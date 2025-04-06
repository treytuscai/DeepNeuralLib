'''inception_ops.py
Fundamental Inception Net operations
YOUR NAMES HERE
CS 444: Deep Learning
Project 2: Branch Neural Networks
NOTE: You should NOT be using an tf.nn.conv2d, Keras, or any other "high level" functions here.
'''
import tensorflow as tf

def conv_1x1(x, filters):
    '''Performs 1x1 convolution to the input image `x` with the filters `filters`. This function should use the matrix
    multiplication approach.

    Parameters:
    -----------
    x: tf.constant. tf.float32s. shape=(Iy, Ix, n_chans).
        The input image.
    filters: tf.constant. tf.float32s. shape=(n_chans, K).
        The convolution filters. `n_chans` is the number of chans in the input, `K` is the number of filters.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(Iy, Ix, K).
        The input image filtered by each of the K filters.

    NOTE: You should use 100% TensorFlow to implement this.
    '''
    Iy, Ix, n_chans = x.shape
    n_chans, n_units = filters.shape

    x_reshaped = tf.reshape(x, [Iy * Ix, n_chans]) # shape = (Iy * Ix, n_chans)
    conv_output = tf.matmul(x_reshaped, filters) # shape = (Iy * Ix, K)
    conv_output = tf.reshape(conv_output, [Iy, Ix, n_units])  # shape = (Iy, Ix, K)
    
    return conv_output

def conv_1x1_batch(x, filters, strides=1):
    '''Performs 1x1 convolution to the mini-batch of input images `x` with the filters `filters`.
    This function should use the BATCH matrix multiplication approach.

    Parameters:
    -----------
    x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
        A mini-batch of images.
    filters: tf.constant. tf.float32s. shape=(n_chans, K).
        The convolution filters. `n_chans` is the number of chans in the input, `K` is the number of filters.
    strides: int.
        Spatial stride to apply to the net input (once you compute it). Applies to both the horizontal and vertical dims.
        NOTE: This is not used in InceptionNet but it is in the upcoming ResNet so let's just add support now.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(B, Iy/stride, Ix/stride, K).
        The net input — input images filtered by each of the K filters.

    NOTE: You should use 100% TensorFlow to implement this, since you will call this in your InceptionNet.
    '''
    B, Iy, Ix, n_chans = x.shape
    n_chans, n_units = filters.shape

    x_reshaped = tf.reshape(x, [B, Iy * Ix, n_chans]) # shape = (B, Iy * Ix, n_chans)
    conv_result = tf.matmul(x_reshaped, filters) # shape = (B, Iy * Ix, K)
    conv_result = tf.reshape(conv_result, (B, Iy, Ix, n_units)) # shape = (B, Iy, Ix, K)

    if strides > 1:
        conv_result = conv_result[:, ::strides, ::strides, :]

    return conv_result


def global_avg_pooling_2d(x):
    '''Performs 2D global average pooling on the mini-batch of images `x`.

    Parameters:
    -----------
    x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K).
        A mini-batch of images with `K` channels.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(B, K).
        The activations averaged across space in each of the K filters.

    NOTE: You should use 100% TensorFlow to implement this, since you will call this in your InceptionNet.
    '''
    return tf.reduce_mean(x, axis=[1, 2])
