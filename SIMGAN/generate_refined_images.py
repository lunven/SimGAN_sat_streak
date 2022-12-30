import numpy as np
import os
import utils.prologue as prologue
import matplotlib.pyplot as plt
import utils.prologue as prologue
from tensorflow import keras
from simgan import *
 #python generate_refined_images.py --o "multi" --i "refined_test.npy" --k "target_test.npy" --m "refiner.h5"

DATAPATH = "../"
def main(args, seed = 123456789):

    img_height=64
    img_width = 64
    img_channels=1
    batch_size=24
    
    #Load the synthetic images, these images will be refined
    s_img = np.load(DATAPATH + args.o +'_test_samples.npy')
    s_img = np.moveaxis(s_img, 1,-1)
    s_img *=255

    #Load the vector describing wether or not there is a streak in the patch
    satellite = np.load(DATAPATH + args.o +'_test_patch_targets.npy')
    a = np.array(satellite)
    ind = np.where(a==1)[0]

    s_imgs = s_img[ind]
    print('Number of synthetic images with streak:{}'.format(s_imgs.shape[0]))

    #load the target image, the one only with a streak
    target_img = np.load(DATAPATH + args.o +'_test_targets.npy')
    target_imgs = target_img[ind]
    target_imgs = np.expand_dims(target_imgs, axis=3)
    print('Target with streaks:{}'.format(target_imgs.shape[0]))

    #this model is the one obtained in simgan_training
    refiner_model = keras.models.load_model('./model/'+args.m,custom_objects={'self_regularization_loss': self_regularization_loss})
    
    datagen = image.ImageDataGenerator(
    preprocessing_function=applications.xception.preprocess_input,
    data_format='channels_last')
    
   
    flow_from_directory_params = {'target_size': (img_height, img_width),
                                'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                'class_mode': None,
                                'batch_size': batch_size}
    flow_params = {'batch_size': batch_size,'shuffle':False}


    synthetic_generator_test = datagen.flow(
        x = s_imgs,
        **flow_params
    )
    
    #A different data generator is used for target images. The difference is that contrary
    #to the synthetic images the target are not maped to [-1,1]
    datagen_ = image.ImageDataGenerator(
        preprocessing_function=None,
        data_format='channels_last')

    target_generator_test = datagen_.flow(
        x = target_imgs,
        **flow_params
    )
    #im is a list containing all refined images of streaks
    im =[]
    #tar is a lit containing all groundtruth of images of im
    tar=[]
    
    def get_image_batch(generator):
        """keras generators may generate an incomplete batch for the last batch"""
        img_batch = generator.next()
        if len(img_batch) != batch_size:
            img_batch = generator.next()

        assert len(img_batch) == batch_size

        return img_batch
    
    
    for i in range(len(synthetic_generator_test)-1):
        synthetic_image_batch = get_image_batch(synthetic_generator_test)
        
        target_image_batch = get_image_batch(target_generator_test)
        
        refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
        
        im.extend(refined_image_batch)
        tar.extend(target_image_batch)

    im = np.array(im)
    tar = np.array(tar)
    
    # to the array of refined images, images where there are no streaks are added 
    # (for the unet). #one out of c images without streaks are kept so that there are around one half 
    #of images with streak and one half without. 
    a = np.array(satellite)
    c = int(np.floor(np.shape(s_img)[0]/np.shape(s_imgs)[0]))
    ind_no = np.where(a==0)[0][::c]
    s_img_no = s_img[ind_no]/255
    s_img_no = (s_img_no*2)-1
    target_no = target_img[ind_no]
    target_no=np.expand_dims(target_no, axis=3)

    tot_refined =[]
    tot_target=[]
    tot_refined.extend(im)
    tot_target.extend(tar)
    tot_refined.extend(s_img_no)
    tot_target.extend(target_no)
    tot_refined = np.array(tot_refined)
    tot_target = np.array(tot_target)
    
    #the array of refined and their corresponding targets are saved
    np.save( "../refined_" +args.i, tot_refined)
    np.save( "../target_"  +args.k, tot_target)

    print('The synthetic images have been refined.')




if __name__ == '__main__':
    main(prologue.get_args())