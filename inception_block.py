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
        super().__init__(blockname, prev_layer_or_block=prev_layer_or_block)

        # Branch 1: 1x1 convolution
        self.branch1_0 = Conv2D1x1(f"{blockname}/branch1_0_conv1x1", branch1_units)
        self.layers.append(self.branch1_0)

        # Branch 2: 1x1 convolution → 3x3 convolution
        self.branch2_0 = Conv2D1x1(f"{blockname}/branch2_0_conv1x1", branch2_units[0])
        self.branch2_1 = Conv2D(f"{blockname}/branch2_1_conv3x3", branch2_units[1], kernel_size=(3,3), wt_init='he', do_batch_norm=True)
        self.layers.extend([self.branch2_0, self.branch2_1])

        # Branch 3: 1x1 convolution → 5x5 convolution
        self.branch3_0 = Conv2D1x1(f"{blockname}/branch3_0_conv1x1", branch3_units[0])
        self.branch3_1 = Conv2D(f"{blockname}/branch3_1_conv5x5", branch3_units[1], kernel_size=(5,5), wt_init='he', do_batch_norm=True)
        self.layers.extend([self.branch3_0, self.branch3_1])

        # Branch 4: 3x3 max pooling → 1x1 convolution
        self.branch4_0 = MaxPool2D(f"{blockname}/branch4_0_maxpool3x3", pool_size=(3,3), padding='SAME')
        self.branch4_1 = Conv2D1x1(f"{blockname}/branch4_1_conv1x1", branch4_units)
        self.layers.extend([self.branch4_0, self.branch4_1])


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
        branch1_out = self.branch1_0(x)

        branch2_out = self.branch2_0(x)
        branch2_out = self.branch2_1(branch2_out)

        branch3_out = self.branch3_0(x)
        branch3_out = self.branch3_1(branch3_out)

        branch4_out = self.branch4_0(x)
        branch4_out = self.branch4_1(branch4_out)

        return tf.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=-1)
