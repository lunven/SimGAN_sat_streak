import numpy as np
import cv2
import os
import operator
import random
import gc
import matplotlib.pyplot as plt
from utils.mosaic import *
from utils.img_processing import *
import utils.prologue as prologue

DATAPATH = "./"
#python generate_synthetic.py --i "synthetic/" --o "multi"

def main(args, seed = 123456789):
    
    path_of_images = args.i
    path_syn = [] #list of strings corresponding to the path of fits images
    # I had a folder containing three folder containing fits images, hence the following two lines
    for j in os.listdir(path_of_images)[1:]:
        for i in os.listdir(path_of_images+ str(j) ):
            path_syn.append(path_of_images+str(j)+'/'+str(i))

    print('Number of fits images in {i}: {j}'.format(i=args.i,j=len(path_syn)))

    satellite = []
    x= []
    y=[]
    nbr = 0

    for w in path_syn:
        print('Image loaded:{i}'.format(i=w))
        im = fits.open(w)
        for s in range(1,31):

            h = im[s].data
            raw_k = scale_image(h.copy())
            unscaled_k = h
            crops_addresses = get_blocks_addresses(raw_k)
            x_ = list(crops_addresses.keys())[0]
            crop = get_block(raw_k, x_, crops_addresses[x_][0])[150:4000,150:2100]
            unscaled_crop = get_block(unscaled_k, x_, crops_addresses[x_][0])[150:4000,150:2100]
            tmp_mask = []


            final_mask = saturated_stars(unscaled_crop) # detect saturated light blob in the image
            mask_dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            final_mask=  cv2.dilate(final_mask,mask_dil, iterations=1) #np.zeros(unscaled_crop.shape)
            h,w = crop.shape
            subh, subw = 64, 64
            for alpha in range(0,h, subh):
                for beta in range(0,w, subw):
                    if (alpha + subh) <= h and (beta+subw) <= w :
                        for k in range(1): # number of samples generated from a single 64x64 patch
                            subcrop = crop[alpha:alpha+subh, beta:beta+subw]
                            x_mirror = random.randint(0,1)
                            y_mirror = random.randint(0,1)
                            if x_mirror == 1:
                                subcrop = subcrop[::-1]
                            if y_mirror == 1:
                                subcrop = subcrop[:,::-1]
                            refcrop = subcrop.copy()
                            
                            subdilation = final_mask[alpha:alpha+subh, beta:beta+subw]
                            star_indices = np.argwhere(subdilation == 1)
                            replacement_value = np.median(subcrop)
                            subcrop[star_indices[:,0], star_indices[:,1]] = replacement_value # remove outlier pixels
                            
                            max_ = np.max(subcrop)
                            min_ = np.min(subcrop)
                            if max_ != min_ : # white patch
                                subcrop = (subcrop - min_) / (max_ - min_) # rescaling
                                
                                y_true = np.zeros(subcrop.shape)
                                mask = np.full(subcrop.shape, 0.)
                            
                                decision = random.random()
                                hs = 0
                                if  decision < 0.01: #a streak is created only 1% of the time
                                    nbr +=1
                                    hs = 1
                                    # STREAK PARAMETRIZATION
                                    numberList = np.arange(2,21)
                                    weight = (4,4,10,10,10,10,10,8,8,3,3,3,3,3,3,3,3,1,1)
                                    s_width= random.choices(numberList, weights=weight, k=1)[0]
                                    
                                    sat_line = np.zeros((64,64,3)).astype(np.uint8)
                                    #The line is defined by two extreme points, located on the borders of the image
                                    # p1 = (p1x,p1y)
                                    # p2 = (p2x, p2y)
                                    
                                    ra1 = random.random()
                                    #in real images the streaks are quite central in the image that is why we constraint the coordinate of
                                    # the second point in an interval that is smaller than (0,64)
                                    if ra1>0.25:
                                        p1x=0
                                        p1y= random.randint(10,50)
                                        ra = random.random()
                                        if ra<1/3:
                                            p2x = 63
                                            p2y = random.randint(0,64)
                                        elif (ra>=1/3)&(ra<2/3):
                                            p2x = random.randint(15,64) 
                                            p2y = 63
                                        else:
                                            p2x = random.randint(15,64)
                                            p2y = 0
                                    elif (ra1>=0.25)&(ra1 < 0.5):
                                        p1x=63
                                        p1y= random.randint(10,50)
                                        ra = random.random()
                                        if ra<1/3:
                                            p2x = 0
                                            p2y = random.randint(0,50)
                                        elif (ra>=1/3)&(ra<2/3):
                                            p2x = random.randint(0,50)
                                            p2y = 63
                                        else:
                                            p2x = random.randint(0,64)
                                            p2y = 0
                                    elif (ra1>=0.5)&(ra1 < 0.75):
                                        p1x=random.randint(10,50)
                                        p1y= 0
                                        ra = random.random()
                                        if ra<1/3:
                                            p2x = 0
                                            p2y = random.randint(15,64)
                                        elif (ra>=1/3)&(ra<2/3):
                                            p2x = random.randint(15,64)
                                            p2y = 63
                                        else:
                                            p2x = 63
                                            p2y = random.randint(5,64)
                                    else:
                                        p1x=random.randint(10,50)
                                        p1y= 63
                                        ra = random.random()
                                        if ra<1/3:
                                            p2x = 0
                                            p2y = random.randint(0,50)
                                        elif (ra>=1/3)&(ra<2/3):
                                            p2x = random.randint(0,55)
                                            p2y = 0
                                        else:
                                            p2x = 63
                                            p2y = random.randint(0,50)
                                            

                                    p1 = (p1x,p1y)
                                    p2 = (p2x,p2y)

                                    sat_line = cv2.line(sat_line,p1 ,p2, (255,255,255), s_width)
                                    
                                    # APPLY THE STREAK IN THE PATCH
                                    h_sat,w_sat,_ = sat_line.shape
                                    
                                    final_synth = cv2.GaussianBlur(sat_line[:,:,0]/255,(s_width*2-1,s_width*2-1),0)
                                    alpha_trans = random.randint(10,20)/100. # opacity of the streak
                                    final_synth = (final_synth / np.max(final_synth))
                                    tmp_mask.append(final_synth)
                                    mask = final_synth
                                    indices = np.argwhere(mask > 0.)
                                    for subx, suby in indices :
                                        subcrop[subx,suby] = max(alpha_trans * mask[subx,suby] + (1-alpha_trans) * subcrop[subx,suby], subcrop[subx,suby])
                                    
                                    y_true = (sat_line[:,:,0] / 255).astype(int)
                                
                                    del sat_line
                                    gc.collect()
                                subcrop = subcrop * (max_ - min_) + min_
                                subcrop[star_indices[:,0], star_indices[:,1]] = refcrop[star_indices[:,0], star_indices[:,1]] # put the blobs in the image back
                                sub_max = np.max(subcrop)
                                sub_min = np.min(subcrop)
                                subcrop = (subcrop - sub_min) / (sub_max - sub_min)
                                
                                

                                # SAVING THE SAMPLES
                                if hs==1: #if there is a streak then the generated image is kept
                                    satellite.append(hs)
                                    x.append([subcrop])
                                    y.append(y_true)
                                if (hs==0)&(random.random()<0.03): #if there is no streak the the generated image is kept only 3% of the time
                                    satellite.append(hs)
                                    x.append([subcrop])
                                    y.append(y_true)

            
            del y_true
            del mask
            del refcrop
            del subcrop
            gc.collect()

            del crop
            del unscaled_crop
            del final_mask
            gc.collect()
        print('Number of generated streaks:{i}'.format(i=nbr))
    
    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    np.save(DATAPATH + args.o + "_samples.npy", np.array(x))
    np.save(DATAPATH + args.o + "_targets.npy", np.array(y))
    np.save(DATAPATH + args.o + "_patch_targets.npy", np.array(satellite))
    

if __name__ == '__main__':
    main(prologue.get_args())
