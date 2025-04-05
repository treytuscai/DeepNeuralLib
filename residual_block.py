'''residual_block.py
Blocks used to make ResNets
YOUR NAMES HERE
CS444: Deep Learning
Project 2: Branch Neural Networks
'''
import tensorflow as tf

import block
from layers import Conv2D
from inception_layers import Conv2D1x1


class ResidualBlock(block.Block):
    '''The Residual Block. Contains two parallel branches:

    Main branch: Two 3x3 Conv2D layers. 1st uses ReLU activation, 2nd uses linear/identity ("temporary") activation

    Residual branch: The input to the block `x`

    The netActs from both branches are summed and ReLU is applied to "close out" the "temporary linear activation
    in the main branch.
    '''
    def __init__(self, blockname, units, prev_layer_or_block, strides=1):
        '''ResidualBlock constructor.

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. Inception1). Used for debugging/printing summary of net.
        units: int.
            Number of units to use in each layer of the block.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        stride: int.
            The convolutional stride (both horizontal and vertical).

        Properties of all branches:
        ---------------------------
        - He initialization where ever possible/appropriate.
        - Batch normalization where ever possible/appropriate.

        Residual branch:
        ----------------
        - There is no explicit layer needed for this pathway under most conditions. However, only when the spatial
        resolution changes compared to the prev block do you need some special sauce — a 1x1 conv layer. Think about
        how to detect a change in spatial resolution. Only if there is one, THEN create a 1x1 conv layer along this
        branch.

        TODO:
        1. Call the superclass constructor to have it store overlapping parameters as instance vars.
        2. Build out the block.

        NOTE:
        - Be careful with how you link up layers with their prev layer association.
        - When naming the layers belonging to the block, prepend the blockname and number which layer in the block
        it is. For example, 'ResBlock1/conv_0'. This will help making sense of the summary print outs when the net is
        compiled.
        '''
        super().__init__(blockname, prev_layer_or_block=prev_layer_or_block)
        self.strides = strides

        # NOTE: Stride only applied to 1st conv bc that will decrease resolution (dont want to do this 2x)
        # Main branch: 2 3x3 convolutions
        self.main_branch_1 = Conv2D(name=blockname+'/main_3x3conv_1',
                                    units=units,
                                    kernel_size=(3, 3),
                                    activation='relu',
                                    prev_layer_or_block=prev_layer_or_block,
                                    wt_init='he',
                                    strides=strides,
                                    do_batch_norm=True)
        self.main_branch_2 = Conv2D(name=blockname+'/main_3x3conv_2',
                                    units=units,
                                    kernel_size=(3, 3),
                                    activation='linear',  # NOTE: This is linear
                                    prev_layer_or_block=self.main_branch_1,
                                    wt_init='he',
                                    do_batch_norm=True)

        self.layers = [self.main_branch_1, self.main_branch_2]

        # We drop spatial resolution / change num filters so need 1x1 conv special sauce
        if self.strides != 1:
            self.skip_branch = Conv2D1x1(name=blockname+'/skip_conv1x1',
                                         units=units,
                                         activation='linear',
                                         prev_layer_or_block=prev_layer_or_block,
                                         do_batch_norm=True,
                                         strides=strides)

            self.layers.append(self.skip_branch)



    def __call__(self, x):
        '''Forward pass through the Residual Block the activations `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            netActs in the mini-batch from a prev layer/block. K1 is the number of channels/units in the PREV layer or
            block.

        Returns:
        --------
        tf.constant. tf.float32s.
            Activations produced by the Residual block. If the spatial resolution does not change, the output shape
            will be: shape=(B, Iy, Ix, K1). If the spatial resolution does change, the output shape
            will be: shape=(B, Iy/2, Ix/2, K), where K is the number of units in the current block.

        TODO:
        1. Forward pass through the main branch.
        2. If the spatial resolution did not change in the curr block, sum the input with the main branch output and
        tack on a ReLU.
        If there was a resolution change, apply the "special sauce" before doing step 2 (otherwise you will have shape
        issues!).
        '''
        net_act = self.main_branch_1(x)
        net_act = self.main_branch_2(net_act)

        if self.strides != 1:
            # We have a stride >1 which means we change spatial dims and channels (new block). Need 1x1 conv
            # no skip connection / only main branch
            x = self.skip_branch(x)

        # Close the 2nd conv layer with RELU
        net_act = tf.nn.relu(net_act + x)
        return net_act

    def __str__(self):
        '''Custom Residual Block toString method.'''
        if self.strides != 1:
            num_layers = len(self.layers) - 1
        else:
            num_layers = len(self.layers)

        string = self.blockname + ':'
        for l in reversed(range(num_layers)):
            string += '\n\t' + self.layers[l].__str__()

        if self.strides != 1:
            string += '\n\t-->' + self.layers[-1].__str__() + '-->'

        return string
