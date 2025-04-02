'''inception_block.py
The Inception Block
YOUR NAMES HERE
CS444: Deep Learning
Project 2: Branch Neural Networks
'''
import tensorflow as tf

import block
from layers import Conv2D, MaxPool2D
from inception_layers import Conv2D1x1


class InceptionBlock(block.Block):
    '''The Inception Block, the core building block of Inception Net. It consists of 4 parallel branches that come
    together in the end:

    Branch 1: 1x1 convolution.

    Branch 2: 1x1 convolution → 3x3 2D convolution

    Branch 3: 1x1 convolution → 5x5 2D convolution

    Branch 4: 3x3 max pooling → 1x1 convolution
    '''
    def __init__(self, blockname, branch1_units, branch2_units, branch3_units, branch4_units, prev_layer_or_block):
        '''InceptionBlock constructor.

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        branch1_units: int. B1.
            Number of units in Branch 1.
        branch2_units: tuple of ints. (B2_0, B2_1).
            Number of units in the 1x1 conv layer (B2_0) and 2D conv layer (B2_1) in Branch 2.
        branch3_units: tuple of ints. (B3_0, B3_1).
            Number of units in the 1x1 conv layer (B3_0) and 2D conv layer (B3_1) in Branch 3.
        branch4_units: int. B4.
            Number of units in Branch 4.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

        Properties of all branches:
        ---------------------------
        - ReLU used in all convolutional layers (both regular and 1x1 kinds).
        - He initialization where ever possible/appropriate.
        - Batch normalization where ever possible/appropriate.

        TODO:
        1. Call the superclass constructor to have it store overlapping parameters as instance vars.
        2. Build out the block. How you order layers in self.layers does not *really* matter, which you should do so
        in a way that makes sense to you to make potential debugging easier.

        NOTE:
        - Be careful with how you link up layers with their prev layer association.
        - The max pooling layer needs to use same padding otherwise you will run into shape issues.
        - The max pooling layer uses stride 1.
        - When naming the layers belonging to the block, prepend the blockname and number which layer in the block
        it is. For example, 'Inception1/conv_0'. This will help making sense of the summary print outs when the net is
        compiled.
        '''
        pass


    def __call__(self, x):
        '''Forward pass through the Inception Block the activations `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K_prev).
            netActs in the mini-batch from a prev layer/block. K_prev is the number of channels/units in the PREV layer
            or block.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, B1+B2_1+B3_1+B4).
            Activations produced by each Inception block branch, concatenated together along the neuron dimension.
            B1, B2_1, B3_1, B4 refer to the number of neurons at the end of each of the 4 branches.
        '''
        pass
