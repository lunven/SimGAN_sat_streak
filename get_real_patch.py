import csv
import pandas as pd
import os 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import utils.prologue as prologue

#python simgan_training.py --i "images/" 
def main(args):
    path_of_real_images = args.i 
    track = []
    #in the csv can be found the coordinates of the two extreme points of the streak
    df = pd.read_csv('labels.csv', header = None) 
    df[5] = df[5].str.replace(':','_')


    ind =[]
    for i in os.listdir(path_of_real_images):
        if df.index[df[5]==i].tolist()!=[]:
            ind.append(df.index[df[5]==i].tolist()[0])
            track.append(i)                                              
        else:
            os.remove(path_of_real_images+i)
    print('The number of images with streaks is: {i}'.format(i = len(ind)))


    sub_ = []
    for k in range(0, len(ind)):
        
        im = Image.open(path_of_real_images+ track[k])
        im = np.array(im).transpose()
        #extract coordinates of the two extreme points
        x1 =df.iloc[ind[k]][1]
        y1  =df.iloc[ind[k]][2]
        x2=  df.iloc[ind[k]][3]
        y2  =df.iloc[ind[k]][4]

        min_x = min(x1,x2)
        max_x = max(x1,x2)
        min_y = min(y1,y2)
        max_y = max(y1,y2)
        
        #the streak needs to be large enough to fit in a 64x64 image
        if (((max_x - min_x)<150)or((max_y - min_y)<150)):
            pass
        
        else:
            sub = im[min_x:max_x, min_y:max_y]
            
            x_ = sub.shape[0]
            
            # slope of the line
            a = (max_y-min_y)/(max_x-min_x)

            #points of the streak: (vecx[i],vecy[i])
            vec_x  = np.arange(50,x_-50,100)
            l = len(vec_x)-1
            vec_y = [a*vec_x[i] for i in range(len(vec_x))]
            
            vec_x_max = [vec_x[i]+32 for i in range(len(vec_x))]
            vec_x_min = [vec_x[i]-32 for i in range(len(vec_x))]
            

            if (((max_y == y2)and(min_x == x2)) or ((max_y == y1)and(min_x == x1))):
                vec_y_max = [int(vec_y[l-i]+32) for i in range(len(vec_x))]
                vec_y_min = [int(vec_y[l-i]-32) for i in range(len(vec_x))]
            else:
                vec_y_max = [int(vec_y[i]+32) for i in range(len(vec_x))]
                vec_y_min = [int(vec_y[i]-32) for i in range(len(vec_x))]

        

            for i in range(len(vec_x)):
                #sub__: patch of size 64x64 surrounding parts of the streak
                sub__ = sub[vec_x_min[i]:vec_x_max[i],vec_y_min[i]:vec_y_max[i]]
                if (sub__.shape[0]==64)&(sub__.shape[1]==64):
                    sub_.append(sub__)
    
    print('Number of patches containing real streaks and background:{i}'.format(i = len(sub_)))
    sub_ = np.array(sub_)
    np.save('real_images',sub_)


if __name__ == '__main__':
    main(prologue.get_args())
            