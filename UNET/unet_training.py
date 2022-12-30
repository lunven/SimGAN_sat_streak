import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from unet import *
import utils.prologue as prologue

# python unet_training.py --i "refined_l=1e-6small_.npy" --o "target_l=1e-6_.npy" --l 1e-4 --v 500 --n 1


def main(args, seed = 123456789):
    
    img_size = (64, 64)
    num_classes = 2 #a background class and a streak one
    batch_size = 32
    
    data_dir= "../"+args.i
    data_dir_tar = "../"+args.o
    input_img = np.load(data_dir)[0:600]
    #this lign is necessary to map the pixel values from [-1,1] to  [0,1]
    input_img=((input_img+1)*127.5)/255

    target_img = np.load(data_dir_tar)[0:600]
    target_img = target_img.squeeze()
    print('Number of pairs: {i}'.format(i=target_img.shape[0]))
    import random


    # Split our img paths into a training and a validation set
    val_samples = args.v
    random.Random(1337).shuffle(input_img)
    random.Random(1337).shuffle(target_img)
    train_input_img_paths = input_img[:-val_samples]
    train_target_img_paths = target_img[:-val_samples]
    val_input_img_paths = input_img[-val_samples:]
    val_target_img_paths = target_img[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = Streak_batch(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = Streak_batch(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    
    model = get_model(img_size, num_classes)

    #keep track of the loss at each epoch
    from keras.callbacks import History 
    history = History()


    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    opt = keras.optimizers.Adam(learning_rate=args.l)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint("synthetic_seg.h5", save_best_only=True)
        , history
    ]

    # Train the model, doing validation at the end of each epoch.
    model.fit(train_gen, batch_size=batch_size, epochs=args.n, validation_data=val_gen, callbacks=callbacks)
    
    #plot and save the training and validation losses
    train_loss= history.history['loss'][1:]
    validation_loss = history.history['val_loss'][1:]
    n = np.arange(len(train_loss))                                            
    plt.plot(n,train_loss)
    plt.xlabel('epoch')
    plt.plot(n,validation_loss)
    plt.legend(['training loss','validation loss'])
    plt.savefig(args.k[:-2]+'png')
     
    #save the model 
    model.save('model/'+args.k,save_format='h5')

if __name__ == '__main__':
    main(prologue.get_args())