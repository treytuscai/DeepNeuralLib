'''block.py
Defines the parent Block class and VGG blocks
YOUR NAMES HERE
CS444: Deep Learning
'''
from layers import Conv2D, MaxPool2D, Dropout, Dense


class Block:
    '''The `Block` parent class and specifies functionality shared by all blocks. All blocks inherit from this class.'''
    def __init__(self, blockname, prev_layer_or_block):
        '''Block constructor.

        Parameters:
        -----------
        blockname: str.
            Human-readable name for a block (VGGConvBlock_0, VGGConvBlock_1, etc.). Used for debugging and printing
            summary of net.
        prev_layer_or_block: Layer or Block object.
            Reference to the Layer or Block object that is beneath the current Layer object. `None` if there is no
            preceding layer or block.
            Examples VGG6: VGGConvBlock_0 → VGGConvBlock_1 → Flatten → VGGDenseBlock_0 → Dense
                The VGGConvBlock_1 block object has `prev_layer_or_block=VGGConvBlock_1` block.
                The VGGDenseBlock_0 block object has `prev_layer_or_block=Flatten` layer.

        TODO:
        1. Define instance vars for parameters.
        2. Create an instance variable for empty list called layers. It must be called self.layers for this class to
        work as expected. This list will store a reference to all specific layers that belong to the block. This is
        handled by specific blocks (i.e. child classes of this class) so it should remain empty here.
        '''
        pass

    def get_prev_layer_or_block(self):
        '''Returns a reference to the Layer object that represents the layer/block below the current one.

        This method is provided to you, so you should not need to modify it.
        '''
        return self.prev_layer_or_block

    def get_layer_names(self):
        '''Returns a list of human-readable string names of the layers that belong to this block.

        This method is provided to you, so you should not need to modify it.
        '''
        names = []
        for layer in self.layers:
            names.append(layer.get_name())
        return names

    def get_params(self):
        '''Returns a list of trainable parameters spread out across all layers that belong to this block.

        This method is provided to you, so you should not need to modify it.
        '''
        all_params = []

        for layer in self.layers:
            params = layer.get_params()
            all_params.extend(params)

        return all_params

    def get_wts(self):
        '''Returns a list of trainable weights (no biases/other) spread out across all layers that belong to this block.

        This method is provided to you, so you should not need to modify it.
        '''
        all_wts = []

        for layer in self.layers:
            wts = layer.get_wts()

            if wts is not None:
                all_wts.append(wts)

        return all_wts

    def get_mode(self):
        '''Gets the mode of the block (i.e. training, not training). Since this is always the same in all layers,
        we use the first layer in the block as a proxy for all of them.

        This method is provided to you, so you should not need to modify it.
        '''
        return self.layers[0].get_mode()

    def set_mode(self, is_training):
        '''Sets the mode of every layer in the block to the bool value `is_training`.

        This method is provided to you, so you should not need to modify it.
        '''

        for layer in self.layers:
            layer.set_mode(is_training)

    def init_batchnorm_params(self):
        '''Initializes the batch norm parameters in every layer in the block (only should have an effect on them if they
        are configured to perform batch normalization).

        This method is provided to you, so you should not need to modify it.
        '''
        for layer in self.layers:
            layer.init_batchnorm_params()

    def __str__(self):
        '''The toString method that gets a str representation of the layers belonging to the current block. These layers
        are indented for clarity.

        This method is provided to you, so you should not need to modify it.
        '''
        string = self.blockname + ':'
        for layer in reversed(self.layers):
            string += '\n\t' + layer.__str__()
        return string

class VGGConvBlock(Block):
    '''A convolutional block in the VGG family of neural networks. It is composed of the following sequence of layers:

    Conv2D → Conv2D → MaxPool2D

    NOTE:
    - The number of successive conv layers is a configurable option.
    - We leave the option of placing a Dropout layer after the MaxPool2D layer in the block.
    For example:

    Conv2D → Conv2D → MaxPool2D → Dropout
    '''
    def __init__(self, blockname, units, prev_layer_or_block, num_conv_layers=2, pool_size=(2, 2), wt_scale=1e-3,
                 dropout=False, dropout_rate=0.1, wt_init='normal', do_batch_norm=False):
        '''VGGConvBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for a block (VGGConvBlock_0, VGGConvBlock_1, etc.). Used for debugging and printing
            summary of net.
        units: tuple of ints:
            Number of units (i.e. filters) to use in each convolutional layer.
            For example: units[0] would be the number for the 1st conv layer, units[1] would be the number for the 2nd,
            etc.
        num_conv_layers: int.
            Number of 2D conv layers to place in sequence within the block. By default this is 2.
        pool_size. tuple. len(pool_size)=2.
            The horizontal and vertical size of the pooling window.
            These will always be the same. For example: (2, 2), (3, 3), etc.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        dropout: bool.
            Whether to place a dropout layer after the 2D maxpooling layer in the block.
        dropout_rate: float.
            If using a dropout layer, the dropout rate of that layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore Week 1 and until instructed otherwise.
        do_batch_norm. bool:
            Whether to do batch normalization in appropriate layers.
            NOTE: Ignore Weeks 1, 2 and until instructed otherwise.

        TODO:
        1. Call the superclass constructor.
        2. Create the layers that belong to this block and place them (in order) in the self.layers list.

        NOTE:
        1. Keep in mind that the number of adjacent conv layers in the block may be different than the default of 2. So
        do NOT hard-code 2 conv layers. See the parameters.
        2. Just like VGG4, all convolutions are 3x3, the maxpooling stride is always 2, and conv layers use ReLU.
        3. When naming the layers belonging to the block, prepend the blockname and number which layer in the block
        it is. For example, if the block is called 'VGGBlock_0', the conv layers might be called 'VGGBlock_0/conv_0'
        and 'VGGBlock_0/conv_1'. This will help making sense of the summary print outs when the net is compiled.
        '''
        super().__init__(blockname, prev_layer_or_block=prev_layer_or_block)

    def __call__(self, x):
        '''Forward pass through the block the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Data samples. K1 is the number of channels/units in the PREV layer or block.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            Activations produced by the output layer to the data.
            K2 is the number of channels/units in the CURR layer or block.
            NOTE: Iy and Ix represent the spatial dims. The actual spatial dims will likely decrease in the block.

        NOTE:
        1. Use the functional API to perform the forward pass through your network!
        2. There is an elegant/short way to forward thru the block involving self.layers... ;)
        '''
        pass


class VGGDenseBlock(Block):
    '''A dense block in the VGG family of neural networks. It is composed of the following sequence of layers:

    Dense → Dropout

    We leave the option of placing multiple Dense (and optionally Dropout) layers in a sequence. For example, both the
    following could happen:

    Dense → Dropout → Dense → Dropout
    Dense → Dense
    '''
    def __init__(self, blockname, units, prev_layer_or_block, num_dense_blocks=1, wt_scale=1e-3, dropout=True,
                 dropout_rate=0.5, wt_init='normal', do_batch_norm=False):
        '''VGGDenseBlock constructor

        Parameters:
        -----------
        blockname: str.
            Human-readable name for a block (VGGDenseBlock_0, etc.). Used for debugging and printing summary of net.
        units: tuple of ints:
            Number of units to use in each dense layer.
            For example: units[0] would be the number for the 1st dense layer, units[1] would be the number for the 2nd,
            etc.
        num_dense_blocks: int.
            Number of sequences of Dense (and optionally Dropout) layers in a sequence (see examples above).
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        dropout: bool.
            Whether to place a dropout layer after each Dense layer in the block.
        dropout_rate: float.
            If using a dropout layer, the dropout rate of that layer. The same in all Dropout layers.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore Week 1 and until instructed otherwise.
        do_batch_norm. bool:
            Whether to do batch normalization in appropriate layers.
            NOTE: Ignore Weeks 1, 2 and until instructed otherwise.

        TODO:
        1. Call the superclass constructor.
        2. Create the layers that belong to this block and place them (in order) in the self.layers list.

        NOTE:
        1. Keep in mind that the number of dense layers in the block may be different than the default of 1. So
        do NOT hard-code 1 dense layers. See the parameters.
        2. Just like VGG4, dense layers in the block use ReLU.
        3. When naming the layers belonging to the block, prepend the blockname and number which layer in the block
        it is. For example, if the block is called 'VGGBlock_0', the conv layers might be called 'VGGBlock_0/dense_0'
        and 'VGGBlock_0/dense_1'. This will help making sense of the summary print outs when the net is compiled.

        '''
        pass

    def __call__(self, x):
        '''Forward pass through the block the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy*Ix*K).
            Net act signal from Flatten layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H).
            Activations produced by the output Dense layer to the data.

        NOTE:
        1. Use the functional API to perform the forward pass through your network!
        2. There is an elegant/short way to forward thru the block involving self.layers... ;)
        '''
        pass
