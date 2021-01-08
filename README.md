Code for the paper Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed.

Abstract: Iterative generative models, such as noise conditional score networks and denoising diffusion probabilistic models, produce high quality samples by gradually denoising an initial noise vector. However, their denoising process has many steps, making them 2-3 orders of magnitude slower than other generative models such as GANs and VAEs. In this paper, we establish a novel connection between knowledge distillation and image generation with a technique that distills a multi-step denoising process into a single step, resulting in a sampling speed similar to other single-step generative models. Our Denoising Student generates high quality samples comparable to GANs on the CIFAR-10 and CelebA datasets, without adversarial training. We demonstrate that our method scales to higher resolutions through experiments on 256x256 LSUN.

We trained models on 4 different datasets: CIFAR-10 at 32x32 resolution, CelebA at 64x64, LSUN Bedroom at 256x256, and LSUN Church_outdoor at 256x256.
These models are made public on [google drive](https://drive.google.com/file/d/1tW5t3W4wqE5f0NXaaiYuFK_2JOBrf9cY/view?usp=sharing)

### Instructions for usage:

First, clone our repository and install the requirements

{DATASET}: should be one of: cifar10, celeba, lsun_bedroom, and lsun_church

#### Sampling from our model:

To sample from our model and save a figure of the images in a directory {FIGURES_DIR}:

`python eval.py figures {FIGURES_DIR} {DATASET} --n_images {N_IMAGES}`

example usage:

`python eval.py figures ./figures cifar10 --n_images 49`

will save a figure of a 7x7 grid of CIFAR-10 images under ./figures 

To sample from our model and save each image under a separate file, in a directory {IMAGES_DIR}:

`python eval.py tofolder {IMAGES_DIR} {DATASET}`

#### To retrain our model:

`python training.py {DATASET} {SAVE_DIR}`

For help, or information on additional arguments run:

python training.py --h, or
python eval.py --h

Information about training and evaluating our model:
evaluation does not support multiple GPUs or TPUs. Sampling with the CPU is supported, and our model does not take long to sample from on the CPU.

When retraining our model, we require at least 2 high end Nvidia GPUs OR Google TPUs. In our experiments, we used TPUv2-8s. 
We highly recommend at least 32GB of RAM and 32GB of Video RAM. If you encounter Out-Of-Memory issues, you may reduce the batch size through the --batch_size argument, but we recommend avoiding this as it would lead to different results. We do not support gradient accumulation.
Before training begins, we sample 1.024M images with a DDIM and save X_T and F_teacher(X_T) as "x_train_{}" and "y_train_{}" respectively. 
Training LSUN models with 1.024M examples (as done in the paper) requires a lot of disk space (~600GB). If you cannot store all 1.024M examples, add the argument `--use_fewer_lsun_examples False` , but we recommend avoiding this as it may hurt reproducibility.
When retraining, you may obtain slightly different results from what was reported since the training data X_T is sampled randomly (using `tf.random.normal`). 


### If you found our work relevant to your research, please cite us:
`
@misc{luhman2021knowledge,  
            title={Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed},  
            author={Eric Luhman and Troy Luhman},  
            year={2021},  
            eprint={2101.02388},  
            archivePrefix={arXiv},  
            primaryClass={cs.LG}  
}
`
