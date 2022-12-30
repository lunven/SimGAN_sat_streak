import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image

import matplotlib
from matplotlib import pyplot as plt

from itertools import groupby
from skimage.util import montage as montage2d

"""
Module to plot a batch of images along w/ their corresponding label(s)/annotations and save the plot to disc.

Use cases:
Plot images along w/ their corresponding ground-truth label & model predicted label,
Plot images generated by a GAN along w/ any annotations used to generate these images,
Plot synthetic, generated, refined, and real images and see how they compare as training progresses in a GAN,
etc...
"""


def plot_batch(image_batch, figure_path, label_batch=None):
    
    all_groups = {label: montage2d(np.stack([img[:,:,0] for img, lab in img_lab_list],0)) 
                  for label, img_lab_list in groupby(zip(image_batch, label_batch), lambda x: x[1])}
    fig, c_axs = plt.subplots(1,len(all_groups), figsize=(len(all_groups)*4, 8), dpi = 600)
    for c_ax, (c_label, c_mtg) in zip(c_axs, all_groups.items()):
        c_ax.imshow(c_mtg, cmap='bone')
        c_ax.set_title(c_label)
        c_ax.axis('off')
    fig.savefig(os.path.join(figure_path))
    plt.close()

"""
Module implementing the image history buffer described in `2.3. Updating Discriminator using a History of
Refined Images` of https://arxiv.org/pdf/1612.07828v1.pdf.

"""
class ImageHistoryBuffer(object):
    def __init__(self, shape, max_size, batch_size):
        """
        Initialize the class's state.

        :param shape: Shape of the data to be stored in the image history buffer
                      (i.e. (0, img_height, img_width, img_channels)).
        :param max_size: Maximum number of images that can be stored in the image history buffer.
        :param batch_size: Batch size used to train GAN.
        """
        self.image_history_buffer = np.zeros(shape=shape)
        self.max_size = max_size
        self.batch_size = batch_size

    def add_to_image_history_buffer(self, images, nb_to_add=None):
        """
        To be called during training of GAN. By default add batch_size // 2 images to the image history buffer each
        time the generator generates a new batch of images.

        :param images: Array of images (usually a batch) to be added to the image history buffer.
        :param nb_to_add: The number of images from `images` to add to the image history buffer
                          (batch_size / 2 by default).
        """
        if not nb_to_add:
            nb_to_add = self.batch_size // 2

        if len(self.image_history_buffer) < self.max_size:
            np.append(self.image_history_buffer, images[:nb_to_add], axis=0)
        elif len(self.image_history_buffer) == self.max_size:
            self.image_history_buffer[:nb_to_add] = images[:nb_to_add]
        else:
            assert False

        np.random.shuffle(self.image_history_buffer)

    def get_from_image_history_buffer(self, nb_to_get=None):
        """
        Get a random sample of images from the history buffer.

        :param nb_to_get: Number of images to get from the image history buffer (batch_size / 2 by default).
        :return: A random sample of `nb_to_get` images from the image history buffer, or an empty np array if the image
                 history buffer is empty.
        """
        if not nb_to_get:
            nb_to_get = self.batch_size // 2

        try:
            return self.image_history_buffer[:nb_to_get]
        except IndexError:
            return np.zeros(shape=0)

def refiner_network(input_image_tensor):
    """
    The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.

    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    def resnet_block(input_features, nb_features=64, nb_kernel_rows=3, nb_kernel_cols=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.

        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.

        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
      
        y = layers.Convolution2D(nb_features, (nb_kernel_rows, nb_kernel_cols), padding='same')(input_features)
        y = layers.Activation('relu')(y)
        y = layers.Convolution2D(nb_features, (nb_kernel_rows, nb_kernel_cols), padding='same')(y)
        print('Fin res: {i}'.format(i=y.shape))
        #y = layers.merge([input_features, y], mode='sum')
        y = layers.Add()([input_features,y])
        return layers.Activation('relu')(y)
    
    print('Entrée: {i}'.format(i=input_image_tensor.shape))
    # an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
    x = layers.Convolution2D(64, (3, 3), padding='same', activation='relu')(input_image_tensor) #64x16
    print('Avant conv: {i}'.format(i=x.shape))
    # the output is passed through 4 ResNet blocks
    for _ in range(4):
        x = resnet_block(x)

    # the output of the last ResNet block is passed to a 1 × 1 convolutional layer producing 1 feature map
    # corresponding to the refined synthetic image
    return layers.Convolution2D(1, 1, 1, padding='same', activation='tanh')(x)

def discriminator_network(input_image_tensor):
    """
    The discriminator network, Dφ, contains 5 convolution layers and 2 max-pooling layers.

    :param input_image_tensor: Input tensor corresponding to an image, either real or refined.
    :return: Output tensor that corresponds to the probability of whether an image is real or refined.
    """
    x = layers.Convolution2D(96, (3, 3),strides=(2, 2), padding='same', activation='relu')(input_image_tensor)
    x = layers.Convolution2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), padding='same', strides=(1, 1))(x)
    x = layers.Convolution2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
    x = layers.Convolution2D(32, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
    x = layers.Convolution2D(2,(1, 1), padding='same', strides=(1, 1), activation='relu')(x)

    # here one feature map corresponds to `is_real` and the other to `is_refined`,
    # and the custom loss function is then `tf.nn.sparse_softmax_cross_entropy_with_logits`
    return layers.Reshape((-1, 2))(x)


# define custom l1 loss function for the refiner
def self_regularization_loss(y_true, y_pred):
    delta = 0.0000001  # FIXME: need to figure out an appropriate value for this
    return tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))


# define custom local adversarial loss (softmax for each image section) for the discriminator
# the adversarial loss function is the sum of the cross-entropy losses over the local patches
def local_adversarial_loss(y_true, y_pred):
    # y_true and y_pred have shape (batch_size, # of local patches, 2), but really we just want to average over
    # the local patches and batch size so we can reshape to (batch_size * # of local patches, 2)
    y_true = tf.reshape(y_true, (-1, 2))
    y_pred = tf.reshape(y_pred, (-1, 2))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    return tf.reduce_mean(loss)