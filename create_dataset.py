import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os 
from utils import get_strategy, make_runfn, float_to_image, show_images
from IPython import display
from models import Onestep_Model
from tqdm import tqdm

'''
    Note: when we ran our experiment for the CelebA dataset, we used a quadratic progression.
    This is different from what was used in [1]. [1] reported an FID of 6.53 when using a linear progression
    However we found that using the quadratic progression led to an FID score closer to what was reported in [1].
    In our experiments, quadratic was 6.3 and linear was 7.1, so we decided to use the quadratic progression. 

    [1]: https://arxiv.org/abs/2010.02502
'''

def write_numpy_images(data_to_use: str, strategy, datadir: str, original_model_dir: str, batch_size: int, shardsize, num_test_examples, n_examples=1024000):
    
    if 'lsun' in data_to_use:
        use_quadratic = False
        num_timesteps = 50
    else:
        use_quadratic = True
        num_timesteps = 100

    with strategy.scope(): 
        model = Onestep_Model(data_to_use, os.path.join(original_model_dir, 'model_tf2_%s.h5' % data_to_use))

    get_xtm1 = make_runfn(model, strategy, run_ddim_process=True)

    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    res = model.spatialres

    def pyfunc(xtr):
        xtr = tf.random.normal([batch_size, res, res, 3], dtype=tf.float32)
        return tf.cast(xtr, tf.float32)

    if not use_quadratic:
        seq = range(0, 1000,  1000//num_timesteps)
    else:
        seq = (np.linspace( 0, np.sqrt(800), num_timesteps)** 2)
        seq = [int(s) for s in list(seq)]

    seq_next = [-1] + list(seq[:-1])
    nshards = n_examples//shardsize + 1
    print("Creating {} shards of {}^2 images on {} steps".format(nshards, model.spatialres, len(seq)))
    beta_set = tf.linspace(1e-4, 0.02, 1000)
    alpha_set = tf.math.cumprod(1-beta_set)
    starter = 0

    for shardnum in tqdm(range(nshards)):
        if shardnum==0:
            if num_test_examples == 0:
                continue
            else:
                assert num_test_examples%batch_size==0

            xtr = tf.range(num_test_examples)
        else: 
            xtr = tf.range(shardsize)

        ds = tf.data.Dataset.from_tensor_slices((xtr)).batch(batch_size, drop_remainder=True)
        ds = ds.map(pyfunc)

        dataset = strategy.experimental_distribute_dataset(ds)

        X_TR = np.zeros([0, res, res, 3]).astype('float16')
        Y_TR = np.zeros([0, res, res, 3]).astype('uint8')
        for x in dataset:
            inputs = tf.concat(x.values, axis=0)
            bs = inputs.shape[0]//strategy.num_replicas_in_sync
            for i, j in zip(reversed(seq), reversed(seq_next)): 
                index = tf.constant(i, dtype=tf.float32) * tf.ones([bs])

                alpha = alpha_set[i] * tf.ones([bs, 1, 1, 1]) 

                alpha_next = alpha_set[j] if j>=0 else tf.constant(1.0)
                alpha_next = alpha_next * tf.ones([bs, 1, 1, 1]) 
                beta = beta_set[i] * tf.ones([bs, 1, 1, 1]) 

                x = get_xtm1(x, index, alpha, alpha_next)

            outputs = tf.concat(x.values, axis=0)
            if starter == 0:
                show_images(outputs[:6], 5, savepath="./example_teacher_imgs.png")
                starter += 1
            inputs = tf.cast(inputs, tf.float16).numpy()
            outputs = float_to_image(outputs)
            X_TR = np.concatenate((X_TR, inputs), axis=0)
            Y_TR = np.concatenate((Y_TR, outputs), axis=0)
        
        if shardnum==0:
            np.save(os.path.join(datadir, 'x_test'), X_TR)
            np.save(os.path.join(datadir, 'y_test'), Y_TR)
        else:
            np.save(os.path.join(datadir, 'x_train_{}'.format(shardnum-1)), X_TR)
            np.save(os.path.join(datadir, 'y_train_{}'.format(shardnum-1)), Y_TR)
        
        del dataset
        del ds

