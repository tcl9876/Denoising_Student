import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from time import time
import tensorflow_addons as tfa
import os
import pickle
from IPython import display
from models import Onestep_Model
from utils import WarmupSchedule, show_images, get_strategy, float_to_image, load_models_from_gdrive, get_settings
from tqdm import tqdm

def train_pyfunc(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    y = (y / 127.5) -1
    return x, y
    
def load_dataset(datadir, shardnum, strategy, batch_size):
    if shardnum == -1:
        X = np.load(os.path.join(datadir, 'x_test.npy'))
        Y = np.load(os.path.join(datadir, 'y_test.npy'))
    else:
        os.path.join(datadir, 'y_train_%d.npy' % shardnum)
        X = np.load(os.path.join(datadir, 'x_train_%d.npy' % shardnum))
        Y = np.load(os.path.join(datadir, 'y_train_%d.npy' % shardnum))

    xtensor = tf.constant(X)
    ytensor = tf.constant(Y)

    del X
    del Y

    ds = tf.data.Dataset.from_tensor_slices((xtensor, ytensor)).map(train_pyfunc)
    ds = ds.shuffle(batch_size*10).batch(batch_size, drop_remainder=True)
    dataset = strategy.experimental_distribute_dataset(ds)
    return dataset

def make_training_objects(model, optimizer, strategy, batch_size, use_l2):

    with strategy.scope():
        test_loss = tf.keras.metrics.Mean()

    if use_l2:
        def lossfn(y, pred_y):
            return tf.reduce_sum(tf.square(y - pred_y)) / batch_size
    else:
        def lossfn(y, pred_y):
            return tf.reduce_sum(tf.math.abs(y - pred_y)) / batch_size

    #"x" is X_T ~ N(0, I), and "y" is the output F_teacher(X_T)
    #this trainstep minimizes Eqn 11
    @tf.function
    def train_step(x, y):
        def step_fn(x, y):

            with tf.GradientTape() as tape:
                pred_y = model(x, training=True)
                loss = lossfn(y, pred_y)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        strategy.run(step_fn, args=(x, y))

    @tf.function
    def test_step(x, y):
        def test_step_fn(x, y):
            pred_y = model(x)
            loss = lossfn(y, pred_y)
            test_loss(loss)

        strategy.run(test_step_fn, args=(x, y))

    return train_step, test_step, test_loss

def get_test_loss(testdataset, test_step, test_loss):
    test_loss.reset_states()
    for x, y in testdataset:
        test_step(x, y)

    print("Test Loss %f" % test_loss.result())
    return test_loss.result()

def train(data_to_use: str, savedir, datadir, original_model_dir, devices_to_use, batch_size, use_xla, use_fewer_lsun_examples=False):
    
    tf.compat.v1.logging.set_verbosity("ERROR")

    data_to_use = data_to_use.lower()
    assert data_to_use in ["cifar10", "celeba", "lsun_church", "lsun_bedroom"]

    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    if not devices_to_use:
        devices_to_use = None
    
    if original_model_dir == "download_from_web":
        original_model_dir = './Original_Models'
        load_models_from_gdrive("./", True) #automatically makes folder called "./Original_Models"

    strategy, machine_type = get_strategy(devices_to_use)
    if machine_type == 'CPU':
        raise RuntimeError("You cant train on the CPU.")
    elif machine_type == 'GPU':
        if use_xla != "False":
            tf.config.optimizer.set_jit(True)
        
    s = get_settings(data_to_use)
    if batch_size == 'default':
        batch_size = s["batch_size"]
    else:
        batch_size = int(batch_size)
    adambeta1 = s["adambeta1"]
    adambeta2 = s["adambeta2"]
    adameps = s["adameps"]
    average_decay = s["average_decay"]
    learnrate, warmupsteps = s["lr"]
    lr = WarmupSchedule(learnrate, warmupsteps)
    epochs = s["epochs"]

    if data_to_use=="cifar10":
        use_l2 = False
    else:
        use_l2 = True

    if 'lsun' in data_to_use and use_fewer_lsun_examples:
        n_examples = 204800

    else:
        n_examples = 1024000
        
    if datadir == "create":
        datadir = "./datasets"
        from create_dataset import write_numpy_images
        n_test_examples = s["n_test_examples"]
        shardsize = s["shardsize"]
        write_numpy_images(data_to_use=data_to_use, strategy=strategy, datadir=datadir, 
        original_model_dir=original_model_dir, shardsize=shardsize,
        batch_size=2*batch_size, num_test_examples=n_test_examples, n_examples=n_examples)
        #batch size for creating dataset is set to twice the training batch size for faster sampling.
    elif not os.path.isdir(datadir):
        raise RuntimeError("Data directory not found")
    elif len(os.listdir(datadir)) ==0:
        raise RuntimeError("Data directory is empty")
    else:
        print("Using the dataset in {}".format(datadir))

    os.mkdir("./model_samples")

    testdataset = load_dataset(datadir, -1, strategy, batch_size = batch_size)

    save_path = os.path.join(savedir, '{}_nonema_model.h5'.format(data_to_use))
    opt_path = os.path.join(savedir, '{}_optimizer.p'.format(data_to_use))

    with strategy.scope():
        model = Onestep_Model(data_to_use, os.path.join(original_model_dir, 'model_tf2_%s.h5' % data_to_use)) 
        optimizer = tf.keras.optimizers.Adam(lr, beta_1=adambeta1, beta_2=adambeta2, epsilon=adameps)
        optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=average_decay)

    train_step, test_step, test_loss = make_training_objects(model, optimizer, strategy, batch_size, use_l2)

    with strategy.scope():
        z = tf.random.normal([1, model.spatialres, model.spatialres, 3])
        model(z)

    if os.path.isfile(save_path) and os.path.isfile(opt_path):
        continue_training = input("There is an existing model and optimizer. Continue training from here? y/n")
        if continue_training.lower() == 'y':
            model.load_weights(save_path)

            with open(opt_path, 'rb') as f:
                opt_weights = pickle.load(f)
            
            train_step(z, z)
            optimizer.set_weights(opt_weights)

    print("Current Optimizer iterations: %d" % optimizer.iterations.numpy())

    ema_save_path = os.path.join(savedir, '{}_ema_model.h5'.format(data_to_use))

    lowest_ema_loss = 100000.0
    nshards = len(os.listdir(datadir))//2 - 1
    print("Total number of shards is {} shards".format(nshards))
    

    s = time()

    for ep in range(epochs):
        
        for i, shard in enumerate(tqdm(range(nshards))):
            
            dataset = load_dataset(datadir, shard, strategy, batch_size = batch_size)

            for x, y in dataset:
                train_step(x, y) 
                    
            del dataset

            if (i+1)%80==0 or i == nshards-1 :
                print("Optimizer iterations %d,  Time %ds" % (optimizer.iterations.numpy(), time()-s))

                model.save_weights(save_path)
                opt_weights = optimizer.get_weights()
                with open(opt_path, 'wb') as f:
                    pickle.dump(opt_weights, f, pickle.HIGHEST_PROTOCOL)

                optimizer.assign_average_vars(model.variables)

                with tf.device("/CPU:0"):
                    images = model(tf.random.normal([6, model.spatialres, model.spatialres, 3]))
                
                show_images(images, 5, savepath="./model_samples/samples_at_it_{}.png".format(optimizer.iterations.numpy()))

                ema_tl = get_test_loss(testdataset, test_step, test_loss) 
                #test loss early stopping for the EMA model.
                if ema_tl < lowest_ema_loss:
                    print("overwriting ema...")
                    model.save_weights(ema_save_path)
                    lowest_ema_loss = ema_tl

                model.load_weights(save_path)
    
    print("Training is completed. The name of the final trained model is {}".format(ema_save_path))
    print("The model was trained for a maximum of {} iterations".format(optimizer.iterations.numpy()))

if __name__ == '__main__':
    
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_to_use", type=str, help="which model to retrain. should be one of ['cifar10', 'celeba', 'lsun_bedroom', 'lsun_church'] ")
    parser.add_argument("savedir", type=str, help="the directory to save model to.")
    parser.add_argument("--datadir", type=str, default="create", help="the directory where the data is located. by default it will be created in ./datasets ")
    parser.add_argument("--original_model_dir", type=str, default="download_from_web", help="the directory where the original models are located. by default it will get them from the web")
    parser.add_argument("--devices", nargs="*", default=[], help="which devices to train on.")
    parser.add_argument("--batch_size", default='default', help="batch size to train model on. recommend keeping at default, or else reproducibility may be affected.")
    parser.add_argument("--xla", action='store_false', help="whether to use XLA, True/False.")
    parser.add_argument("--use_fewer_lsun_examples", action='store_true', help="when training LSUN, whether to use all 1.024M examples. If you are short on disk space, set to True. This may affect reproducibility if set to True.")
    args = parser.parse_args()
    
    train(args.data_to_use, args.savedir, args.datadir, args.original_model_dir, args.devices, args.batch_size, args.xla)
