from imageio import imwrite
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from utils import show_images, slerp, make_batch_of_images, float_to_image, load_models_from_gdrive
from models import Onestep_Model

#note: these functions do NOT support multi-gpu or TPU. Will either run on one GPU if available, otherwise CPU
def interpolation_experiment(model, device, n_images=11, savepath=None):
    tset = [i/(n_images-1) for i in range(n_images)]
    assert min(tset) == 0. and max(tset) == 1.
    z1, z2 = tf.split(tf.random.normal([2, model.spatialres, model.spatialres, 3]), 2)
    z_in = tf.concat([slerp(z1, z2, tset[i]) for i in range(n_images)], axis=0)
    

    with tf.device(device):
        images = model(z_in)

    show_images(images, dims=[1, n_images], savepath=savepath)

def getmodelimages(model, device, bs):
    z = tf.random.normal([bs, model.spatialres, model.spatialres, 3])
    with tf.device(device):
        images = model(z)
    return images

def getmodel(data_to_use, denoising_student_dir):
    model = Onestep_Model(data_to_use, None)
    model(tf.random.normal([1, model.spatialres, model.spatialres, 3]))
    model.load_weights(os.path.join(denoising_student_dir, '%s_ema_model.h5' % data_to_use))
    return model

def write_images_to_folder(model, device, write_dir, batch_size=20, n_images=20):
    '''
    this function can be used to write 50k images for metrics (IS and FID)
    metrics are each calculated against 50k training images from cifar10 and (FID only) celeba.
    both IS and FID use the official code provided by [1] and [2] respectively. 
    celeba images are prepared in same manner as [3]
    [1] https://arxiv.org/abs/1606.03498 
    [2] https://arxiv.org/abs/1706.08500
    [3] https://arxiv.org/abs/2010.02502
    '''
    if write_dir is not None:
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)
    
    n_batches = n_images//batch_size
    remaining_samples = n_images - batch_size*n_batches
    n_batches += 1
    n_written = 0

    for i in tqdm(range(n_batches)):
        if i == n_batches - 1:
            bs = remaining_samples
        else:
            bs = batch_size
        if bs==0:
            continue
        images = getmodelimages(model, device, bs)
        images = float_to_image(images)

        if write_dir is not None:
            for img in images:
                imgpath = os.path.join(write_dir, 'images{}.png'.format(str(n_written)))
                imwrite(imgpath, img)        
                n_written += 1

    return n_written == n_images

def get_uncurated_samples(data_to_use, model_dir, savedir, device, n_images):
    model = getmodel(data_to_use, model_dir)
    images = getmodelimages(model, device, n_images)
    savepath = os.path.join(savedir, '{}_figure_{}.png'.format(data_to_use, len(os.listdir(savedir))))
    dims=[np.ceil(np.sqrt(n_images)), np.ceil(np.sqrt(n_images))]
    scale = min(model.spatialres, 192)//32

    if os.path.isfile(savepath):
        print("There is a file here already. it will be overwritten.")
    show_images(images, scale=scale, savepath=savepath, dims=dims)
    return True
    
def main(action, savedir, data_to_use, n_images, model_dir, batch_size):
    if model_dir == "download_from_web":
        model_dir = './denoising_student_models'
        if not os.path.exists(model_dir):
            load_models_from_gdrive("./", False)
    
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    print("Running on device {}".format(device))
    print("TPU and Multi-GPU setups are not supported in evaluation.")

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if action == 'figures':
        status = get_uncurated_samples(data_to_use, model_dir, savedir, device, n_images)
    elif action == 'tofolder':
        model = getmodel(data_to_use, model_dir)
        status = write_images_to_folder(model, device, savedir, batch_size, n_images)
    else:
        raise NotImplementedError("action must be 'figures' or 'tofolder'. ") 

    if status:
        print("Finished execution properly.")
    

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", type=str, default='figures', help="what action to do. should be either 'figures', 'tofolder'. 'figures' option will create a square figure of images. 'tofolder' option will write each image to a file")
    parser.add_argument("savedir", type=str, help="the directory to save outputs to.")
    parser.add_argument("data_to_use", type=str, help="Which dataset's images to write. should be one of ['cifar10', 'celeba', 'lsun_bedroom', 'lsun_church'] ")
    parser.add_argument("--n_images", type=int, default=20, help="how many images to write.")
    parser.add_argument("--model_dir", type=str, default="download_from_web", help="the directory where the denoising_student_models are located. by default it will get them from the web")
    parser.add_argument("--batch_size", type=int, default=20, help="when using tofolder, batch size to run examples on.")
    
    args = parser.parse_args()
    
    main(args.action, args.savedir, args.data_to_use, args.n_images, args.model_dir, args.batch_size)
