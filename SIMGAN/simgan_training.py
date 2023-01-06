import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import h5py
from simgan import *
import random


import utils.prologue as prologue
#A big part was taken from:https://github.com/mjdietzx/SimGAN



def main(args):
    
    
    label = args.o 
    ######################### Load the synthetic and real images##################################
    #Load the synthetic images
    s_img = np.load("../" + label + '_test_samples.npy')
    s_img = np.moveaxis(s_img, 1,-1)
    s_img *=255

    #Load the real images
    r_imgs = np.load("../" + '/real_images.npy')
    r_imgs = np.expand_dims(r_imgs, axis=3)
    print('Number of real images:{}'.format(r_imgs.shape[0]))


    #Load the vector describing wether or not there is a streak in the patch
    satellite = np.load("../"+label+'_test_patch_targets.npy')
    a = np.array(satellite)
    ind = np.where(a==1)[0]

    s_imgs = s_img[ind]
    print('Number of synthetic images with streak:{}'.format(s_imgs.shape[0]))

    #load the target image, the one only with a streak
    target_img = np.load("../"+label+'_test_targets.npy')
    target_imgs = target_img[ind]
    target_imgs = np.expand_dims(target_imgs, axis=3)
    print('Number of target with streak:{}'.format(target_imgs.shape[0]))
    
    ########################## Definition of parameters #############################################
    nb_steps = args.n # originally 10000, but this makes the kernel time out
    batch_size = args.b
    k_d = 1  # number of discriminator updates per step
    k_g = 2  # number of generative network updates per step
    log_interval = 40
  
    img_height= 64
    img_width=64
    img_channels = 1
    
    ########################## model declaration ####################################################
    
    # define model input and output tensors

    synthetic_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    refined_image_tensor = refiner_network(synthetic_image_tensor)

    refined_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    discriminator_output = discriminator_network(refined_or_real_image_tensor)


    # define models
    refiner_model = models.Model(synthetic_image_tensor, refined_image_tensor, name='refiner')
    discriminator_model = models.Model(refined_or_real_image_tensor, discriminator_output,name='discriminator')

    # combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
    refiner_model_output = refiner_model(synthetic_image_tensor)
    combined_output = discriminator_model(refiner_model_output)
    combined_model = models.Model(synthetic_image_tensor, [refiner_model_output, combined_output],
                                name='combined')

    discriminator_model_output_shape = discriminator_model.output_shape
    
    ########################## training initialization ##################################################
    sgd = optimizers.SGD(learning_rate=args.l)

    refiner_model.compile(optimizer=sgd, loss=self_regularization_loss) #L1 loss

    discriminator_model.compile(optimizer=sgd, loss=local_adversarial_loss) 
    discriminator_model.trainable = False

    combined_model.compile(optimizer=sgd, loss=[self_regularization_loss, local_adversarial_loss]) #the loss minimized will be the sum of the 2 losses
    
    ########################## batch generator ##########################################################
    #the preprocessing function maps images to pixel values between [-1,1]
    datagen = image.ImageDataGenerator(
    preprocessing_function=applications.xception.preprocess_input,
    data_format='channels_last')

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                'class_mode': None,
                                'batch_size': batch_size}
    flow_params = {'batch_size': batch_size}

    synthetic_generator = datagen.flow(
        x = s_imgs,
        **flow_params
    )

    real_generator = datagen.flow(
        x = r_imgs,
        **flow_params
    )


    def get_image_batch(generator):
        """keras generators may generate an incomplete batch for the last batch"""
        img_batch = generator.next()
        if len(img_batch) != batch_size:
            img_batch = generator.next()

        assert len(img_batch) == batch_size

        return img_batch

    # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
    y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
    y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)
    assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)
    batch_out = get_image_batch(synthetic_generator)
    assert batch_out.shape == (batch_size, img_height, img_width, img_channels), "Image Dimensions do not match, {}!={}".format(batch_out.shape, (batch_size, img_height, img_width, img_channels))

   ################################# TRAINING ########################################################
    image_history_buffer = ImageHistoryBuffer((0, img_height, img_width, img_channels), batch_size * 10, batch_size) #size of images, max size of buffer, batch size

    combined_loss = np.zeros(shape=3)
    disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
    disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))
    LOSS_r = []
    LOSS_d =[]
    LOSS_d_r = []


   
    for i in range(nb_steps):
        
        # train the refiner
        for _ in range(k_g * 2):
            # sample a mini-batch of synthetic images
            synthetic_image_batch = get_image_batch(synthetic_generator)
            a = combined_model.train_on_batch(synthetic_image_batch,[synthetic_image_batch, y_real])
            
            
            # update θ by taking an SGD step on mini-batch loss LR(θ)
            combined_loss = np.add(a, combined_loss)
    

        for _ in range(k_d):
            # sample a mini-batch of synthetic and real images
            synthetic_image_batch = get_image_batch(synthetic_generator)
            real_image_batch = get_image_batch(real_generator)
            
            #data augmentation
            if random.random()>0.5:
                real_image_batch = np.flip(real_image_batch,axis=1)
                
            if random.random()>0.5:
                real_image_batch = np.flip(real_image_batch,axis=2)
                
            if random.random()>0.5:
                real_image_batch = np.rot90(real_image_batch,axes=(1,2))
            
            if random.random()>0.5:
                real_image_batch = np.rot90(real_image_batch,axes=(1,2))
            

            # refine the synthetic images w/ the current refiner
            refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)

            # use a history of refined images
            half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
            image_history_buffer.add_to_image_history_buffer(refined_image_batch)

            if len(half_batch_from_image_history):
                refined_image_batch[:batch_size // 2] = half_batch_from_image_history

            # update φ by taking an SGD step on mini-batch loss LD(φ)
            disc_loss_real = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss_real)
            disc_loss_refined = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined),
                                    disc_loss_refined)

        if not i % log_interval:
            print('Step: {} of {}.'.format(i, nb_steps))
            
            synthetic_image_batch = get_image_batch(synthetic_generator)
            
        
            LOSS_r.append(combined_loss / (log_interval * k_g * 2))
            LOSS_d.append(disc_loss_real / (log_interval * k_d * 2))
            LOSS_d_r.append(disc_loss_refined / (log_interval * k_d *2))
            
            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
            disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))
    print('L1 loss of the refiner: {i}:'.format(i = LOSS_r[-1][0] ))
    print('Discriminator loss for real images: {i}:'.format(i = LOSS_d_r[-1] ))
    print('Discriminator loss for refined images: {i}:'.format(i = LOSS_d[-1] ))

    refiner_model.save('./model/refiner.h5')
    discriminator_model.save('./model/discriminator.h5')   

if __name__ == '__main__':
    main(prologue.get_args())
