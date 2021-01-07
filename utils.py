from imageio import imwrite
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import requests
import zipfile

def get_settings(data_to_use):
    if 'lsun' in data_to_use:
        is_bedroom = 'bedroom' in data_to_use
        data_to_use = 'lsun'
    settings = {
        "cifar10":
        {
         "adambeta1": 0.9,
         "adambeta2": 0.98,
         "adameps": 1e-8,
         "average_decay": 0.995,
         "lr": [ 2e-4, 5000],
         "epochs": 25,
         "shardsize": 102400,
         "n_test_examples": 51200,
         "batch_size": 512
        }, 
        "celeba":
        {
         "adambeta1": 0.9,
         "adambeta2": 0.98,
         "adameps": 1e-8,
         "average_decay": 0.995,
         "lr": [ 5e-5, 5000],
         "epochs": 35,
         "shardsize": 25600,
         "n_test_examples": 10240,
         "batch_size": 512
        }, 
        "lsun":
        {
         "adambeta1": 0.98,
         "adambeta2": 0.999,
         "adameps": 1e-8,
         "average_decay": 0.9995,
         "lr": [ 5e-6, 1000],
         "shardsize": 2560,
         "n_test_examples": 2560,
         "batch_size": 32
        }
    }
    if 'lsun' in data_to_use:
        if is_bedroom:
            settings['lsun']['epochs'] = 50
        else:
            settings['lsun']['epochs'] = 40
    return settings[data_to_use]

def show_images(images, scale=5, savepath=None, dims = None):
    if isinstance(images, tf.Tensor):
        if len(images.shape) == 4:
            images = tf.split(images, images.shape[0], axis=0)
            for i in range(len(images)):
                images[i] = tf.squeeze(images[i])
    
    if not isinstance(images[0], np.ndarray):
        for i in range(len(images)):
            images[i] = float_to_image(images[i])
    
    if dims is None:
        m = len(images)//10 + 1
        n = 10
    else:
        m, n = dims

    plt.figure(figsize=(scale*n, scale*m))

    for i in range(len(images)):
        plt.subplot(m, n, i+1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        
    plt.show()

def float_to_image(x):
    x = tf.clip_by_value(x, -1.0, 1.0)
    x = x*127.5 + 127.5
    return x.numpy().astype('uint8')

def slerp(z1, z2, t):
    omega = tf.math.acos(tf.reduce_sum(z1 * z2)/(tf.norm(z1)*tf.norm(z2)))
    a = tf.sin((1-t) * omega)/tf.sin(omega)
    b = tf.sin(t*omega)/tf.sin(omega)
    return a * z1 + b * z2

class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learnrate, warmup_steps):
        super().__init__()
        self.learnrate = learnrate
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = self.learnrate
        arg2 = step * self.learnrate / float(self.warmup_steps)
        return tf.math.minimum(arg1, arg2)
    
class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learnrate, warmup_steps):
        super().__init__()
        self.learnrate = learnrate
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = self.learnrate
        arg2 = step * self.learnrate / float(self.warmup_steps)
        return tf.math.minimum(arg1, arg2)
  
def get_strategy(devices_to_use=None):
    if isinstance(devices_to_use, str):
        if "CPU" in devices_to_use.upper():
            return tf.distribute.get_strategy(), "CPU"
    
    if tf.config.list_physical_devices('GPU'):
        return tf.distribute.MirroredStrategy(devices=devices_to_use), "GPU"
    else:
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=devices_to_use)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            print("All devices: ", tf.config.list_logical_devices('TPU'))
            return tf.distribute.TPUStrategy(resolver), "TPU"
        except:
            return tf.distribute.get_strategy(), "CPU"
        
def make_runfn(model, strategy, run_ddim_process=False):
    if not run_ddim_process:
        @tf.function
        def runmodel(z):
            def replica_fn(z):
                return model(z)
            result = strategy.run(replica_fn, args=(z,))
            return result
    else:
        @tf.function
        def runmodel(xt, index, alpha, alpha_next):
            def replica_fn(xt, index, alpha, alpha_next):
                return model.run_ddim_step(xt, index, alpha, alpha_next)
            result = strategy.run(replica_fn, args=(xt, index, alpha, alpha_next))
            return result
    
    return runmodel
    
def make_batch_of_images(model, runfn, z=None, n_samples=None):
    
    if n_samples is None and z is None:
        raise RuntimeError("Specify input or number of samples.")
    elif z is None:
        z = tf.random.normal([n_samples, model.spatialres, model.spatialres, 3])
    else:
        n_samples = z.shape[0]
    
    #must be a tensor of [N, H, W, 3]
    assert list(z.shape[1:]) == [model.spatialres, model.spatialres, 3] and z.dtype == tf.float32
    
    images = runfn(z)
    images = float_to_image(images)
    return images

def load_models_from_gdrive(targetdir, get_original_models):
    if not os.path.exists(targetdir):
        os.mkdir(targetdir)
    
    if get_original_models:
        zipped_loc = os.path.join(targetdir, "Original_Models.zip")
        #loads the adapted H5 files of the trained models in "Denoising Diffusion Probabilistic Models" and "Denoising Diffusion Implicit Models"
        drive_id = "1KlUuwAbWqHI0u9FXTb5xqSOIUnOMI7EV"
    else:
        zipped_loc = os.path.join(targetdir, "denoising_student_models.zip")
        #loads our trained generative models.
        drive_id = "1tW5t3W4wqE5f0NXaaiYuFK_2JOBrf9cY"

    download_file_from_google_drive(drive_id, zipped_loc)
    with zipfile.ZipFile(zipped_loc,"r") as zip_ref:
        zip_ref.extractall(targetdir)
    
#download_file_from_google_drive and its helper functions taken from https://stackoverflow.com/a/39225039
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)