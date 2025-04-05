'''resnets.py
Various neural networks in the ResNet family
YOUR NAMES HERE
CS 444: Deep Learning
Project 2: Branch Neural Networks
'''
import tensorflow as tf

import network
from layers import Conv2D, Dense
from inception_layers import GlobalAveragePooling2D
from residual_block import ResidualBlock


def stack_residualblocks(stackname, units, num_blocks, prev_layer_or_block, first_block_stride=1, block_type='residual'):
    '''Creates a stack of `num_blocks` Residual Blocks, each with `units` neurons.

    Parameters:
    -----------
    stackname: str.
        Human-readable name for the current stack of Residual Blocks
    units: int.
        Number of units in each block in the stack.
    num_blocks: int.
        Number of blocks to create as part of the stack.
    prev_layer_or_block: Layer (or Layer-like) object.
        Reference to the Layer/Block object that is beneath the first block. `None` if there is no preceding
        layer/block.
    first_block_stride: int. 1 or 2.
        The stride on the 1st block in a stack could EITHER be 1 or 2.
        The stride for ALL blocks in the stack after the first ALWAYS is 1.
    block_type: str.
        Ignore for base project. Option here to help in case you want to build very deep ResNets (e.g. ResNet-50)
        for Extensions, which use 'bottleneck' blocks.

    Returns:
    --------
    Python list.
        list of Residual Blocks in the current stack.

    NOTE: To help keep stacks, blocks, and layers organized when printing the summary, modify each block name by
    preprending the stack name to which it belongs. For example, if this is stack_1, call the first two blocks
    'stack_1/block_1' and 'stack_1/block_2'.
    '''
    pass


class ResNet(network.DeepNetwork):
    '''ResNet parent class
    '''
    def __call__(self, x):
        '''Forward pass through the ResNet with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.

        Hint: you are storing all layers/blocks sequentially in self.layers and there are NO skip connections acrossblocks ;)
        '''
        pass

    def summary(self):
        '''Custom toString method for ResNets'''
        print(75*'-')
        for layer in reversed(self.layers):
            print(layer)


class ResNet8(ResNet):
    '''The ResNet8 network. Here is an overview of its structure:

    Conv2D → 3xResidualBlocks → GlobalAveragePooling2D → Dense

    Layer/block properties:
    -----------------------
    - Conv layer: 3x3 kernel size. ReLU activation. He initialization (always!). Uses batch norm.
    - ResidualBlocks: 1st block has stride of 1, the others have stride 2.
    - Dense: He initialization (always!)

    '''
    def __init__(self, C, input_feats_shape, filters=32, block_units=(32, 64, 128), reg=0):
        '''ResNet8 constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: int.
            Number of filters in the 1st 2D conv layer.
        block_units: tuple of ints.
            Number of filters in each residual block.
        reg: float.
            Regularization strength.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the ResNet network. Use of self.layers for organizing layers/blocks.
        3. Remember that although Residual Blocks have parallel branches, the macro-level ResNet layers/blocks
        are arranged sequentially.

        NOTE:
        - To make sure you configure everything correctly, make it a point to check every keyword argment in each of
        the layers/blocks.
        - The only requirement on your variable names is that you MUST name your output layer `self.output_layer`.
        - Use helpful names for your layers and variables. You will have to live with them!
        '''
        pass


class ResNet18(ResNet):
    '''The ResNet18 network. Here is an overview of its structure:

    Conv2D → 4 stacks of 2 ResidualBlocks → GlobalAveragePooling2D → Dense

    Layer/block properties:
    -----------------------
    - Conv layer: 3x3 kernel size. ReLU activation. He initialization (always!). Uses batch norm.
    - Stacks of Residual Blocks: 1st stack blocks in net has stride 1, the first block in the remaining 3 stacks start
    with stride 2. Two blocks per stack.
    - Dense: He initialization (always!)
    '''
    def __init__(self, C, input_feats_shape, filters=64, block_units=(64, 128, 256, 512), reg=0):
        '''ResNet18 constructor

        Parameters:
        -----------
        C: int.
            Number of classes in the dataset.
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            Example: If the input are 32x32 RGB images, input_feats_shape=(32, 32, 3).
        filters: int.
            Number of filters in the 1st 2D conv layer.
        block_units: tuple of ints.
            Number of filters in each residual block.
        reg: float.
            Regularization strength.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the ResNet network. Use of self.layers for organizing layers/blocks.
        3. Remember that although Residual Blocks have parallel branches, the macro-level ResNet layers/blocks
        are arranged sequentially.
        '''
        pass
