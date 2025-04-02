'''inception_layers.py
New layers required for Inception Net
YOUR NAMES HERE
CS 444: Deep Learning
Project 2: Branch Neural Networks
'''
import tensorflow as tf

import layers
from inception_ops import conv_1x1_batch, global_avg_pooling_2d


class Conv2D1x1(layers.Conv2D):
    '''1x1 2D Convolution layer. Inherits from Conv2D so this subclass only needs to override distinct methods/behavior.
    '''
    def __init__(self, name, units, activation='relu', prev_layer_or_block=None, do_batch_norm=True, strides=1):
        '''Conv2D1x1 constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Conv2D1x1_0). Used for debugging/printing summary of net.
        units: ints.
            Number of convolutional filters/units (K).
        activation: str.
            Name of the activation function to apply in the layer.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        do_batch_norm. bool:
            Whether to do batch normalization in this layer.
        strides. int.
            The horizontal AND vertical stride of the convolution operation. These will always be the same.
            By convention, we use a single int to specify both of them.

        TODO: Call the superclass constructor to have it store overlapping parameters as instance vars. Store any unique
        parameters to Conv2D1x1 as instance vars below.

        NOTE: We always want to use He weight initialization :)
        '''
        pass

    def init_params(self, input_shape):
        '''Initializes the Conv2D1x1 layer's weights and biases EXCLUSIVELY using He initialization.

        Parameters:
        -----------
        input_shape: Python list. len(input_shape)=4.
            The anticipated shape of mini-batches of input that the layer will process: (B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.

        NOTE: This is the same as the Conv2D method except for the 2D shape of the weights.
        '''
        N, I_y, I_x, n_chans = input_shape

    def compute_net_input(self, x):
        '''Computes the net input for the current 1x1 2D Convolution layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. K1 is the number of units in the previous layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            The net_in. K2 is the number of units in the current layer.

        TODO:
        1. This layer uses lazy initialization. This means that if the wts are currently None when we enter this method,
        we should call `init_params` to initialize the parameters!
        2. Use the appropriate function that you have already implemented in inception_ops.py to compute the net_in :)
        Do NOT use tf.nn.conv2d here.

        NOTE: Don't forget the bias and to pass along the stride!
        '''
        pass

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Conv2D1x1 layer output({self.layer_name}) shape: {self.output_shape}'


class GlobalAveragePooling2D(layers.Layer):
    '''2D global average pooling layer'''
    def __init__(self, name, prev_layer_or_block=None):
        '''GlobalAveragePooling2D constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. GlobalAveragePooling2D_0).
            Used for debugging/printing summary of net.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

        TODO: Call the superclass constructor, passing in the appropriate information.
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net input using 2D global average pooling.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K).
            netActs from the layer below.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, K).
            The output of the 2D global average pooling operation.

        NOTE: You should defer to your function in inception_ops.
        '''
        pass

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Global Avg Pooling 2D layer output({self.layer_name}) shape: {self.output_shape}'
