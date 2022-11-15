#import relevant libraries
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import  AveragePooling2D, UpSampling2D, BatchNormalization, Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Reshape, Conv2DTranspose, Embedding, Concatenate, Multiply, Add, Lambda
from tensorflow.keras.layers.experimental.preprocessing import Normalization 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import matplotlib.pyplot as plt
import os

import math

image_size = 128
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 3

# optimization
batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

def get_sinusoidal_embedding_channel(t_var):
        #fix the omegas and the number of channels to add
        omegas = 2.0*math.pi*tf.exp(tf.linspace(0.0, math.log(1000), 8))
        
        sinusoid_emb = Lambda(lambda x: tf.concat([tf.sin(x[0] * x[1]), tf.cos(x[0] * x[1])], axis=3), name='Embedding_layer')([omegas, t_var])
        return sinusoid_emb

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = Conv2D(width, kernel_size=1)(x)
        x = BatchNormalization(center=False, scale=False)(x)
        x = Conv2D(
            width, kernel_size=3, padding="same", activation=tf.keras.activations.swish
        )(x)
        x = Conv2D(width, kernel_size=3, padding="same")(x)
        x = Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = Input(shape=(image_size, image_size, 3))
    noise_variances = Input(shape=(1, 1, 1))

    e = get_sinusoidal_embedding_channel(noise_variances)
    e = UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return Model([noisy_images, noise_variances], x, name="residual_unet")
    

def get_network_classifier_guidance(image_size, widths, block_depth, n_attrs):
    noisy_images = Input(shape=(image_size, image_size, 3))
    noise_variances = Input(shape=(1, 1, 1))
    
    e = get_sinusoidal_embedding_channel(noise_variances)
    e = UpSampling2D(size=image_size, interpolation="nearest")(e)
    
    y = Input(shape=(n_attrs, 1, 1))
    y_upsamp = UpSampling2D(size=(int(image_size/n_attrs), image_size), interpolation="nearest")(y)
    y_upsamp = Lambda(lambda x: tf.image.resize_with_crop_or_pad(x[0], x[1].shape[1], x[1].shape[2]), name='y_labels_resize')([y_upsamp, noisy_images])

    
    x = Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = Concatenate()([x, e, y_upsamp])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return Model([noisy_images, noise_variances, y], x, name="residual_unet")