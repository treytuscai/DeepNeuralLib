'''inception_net.py
The Inception Net
YOUR NAMES HERE
CS 444: Deep Learning
Project 2: Branch Neural Networks
'''
import tensorflow as tf

import network
from layers import Conv2D, MaxPool2D, Dropout, Dense
from inception_layers import GlobalAveragePooling2D
from inception_block import InceptionBlock


class InceptionNet(network.DeepNetwork):
    '''Inception Net with the following structure:

    Conv2D → MaxPool2D → 2 InceptionBlocks → MaxPool2D → 3 InceptionBlocks → MaxPool2D → GlobalAveragePool2D → Dropout
    → Dense

    Layer properties:
    -----------------
    - Conv2D layer: 64 filters, 3x3 kernel size. He initialization (always!). Uses batch norm.
    - Max pooling layers: 3x3 window, stride 2. Use 'SAME' padding for all max pooling layers so that the spatial dims
    throughout the net are more predictable/easier to reason about.
    - Dropout: dropout rate of 0.4
    - Dense: He initialization (always!)

    Inception Blocks:
    -----------------

    Each Inception Block has different sets of hyperparameters for the number of units in layers along each branch.
    This large number makes it tedious to make them function parameters, so let's hard code the values for this
    instance of Inception Net. For extensions or other explorations, you can subclass this net class and tweak the
    numbers.

    Here are the number of units in each Inception Block branch in the format:

    [B1, (B2_0, B2_1), (B3_0, B3_1), B4]

    Please see InceptionBlock constructor for a description of what these variables mean.
    Here are the number of units in each Inception Block (i.e. 1st list is for 1st Inception block, 5th is for the last
    Inception block in the net):

    [32, (32, 64), (16, 32), 32]
    [64, (64, 128), (32, 64), 64]
    [64, (96, 128), (32, 64), 64]
    [64, (96, 128), (32, 64), 64]
    [128, (128, 196), (64, 128), 128]
    '''
    def __init__(self, C, input_feats_shape, reg=0):
        '''InceptionNet constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        reg: float.
            Regularization strength.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the InceptionNet network.
        3. Remember that although Inception Blocks have parallel branches, the macro-level InceptionNet layers/blocks
        are arranged sequentially.

        NOTE:
        - To make sure you configure everything correctly, make it a point to check every keyword argment in each of
        the layers/blocks.
        - The only requirement on your variable names is that you MUST name your output layer `self.output_layer`.
        - Use helpful names for your layers and variables. You will have to live with them!
        '''
        pass

    def __call__(self, x):
        '''Forward pass through the InceptionNet with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.
        '''
        pass
