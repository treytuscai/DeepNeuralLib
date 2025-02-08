'''layers.py
Neural network layers (e.g. Dense, Conv2D, etc.) implemented with the low-level TensorFlow API.
YOUR NAMES HERE
CS444: Deep Learning
'''
import tensorflow as tf
from math import pi

class Layer:
    '''Parent class for all specific neural network layers (e.g. Dense, Conv2D). Implements all functionality shared in
    common across different layers (e.g. net_in, net_act).
    '''
    def __init__(self, layer_name, activation, prev_layer_or_block, do_batch_norm=False, batch_norm_momentum=0.99,
                 do_layer_norm=False):
        '''Neural network layer constructor. You should not generally make Layers objects, rather you should instantiate
        objects of the subclasses (e.g. Dense, Conv2D).

        Parameters:
        -----------
        layer_name: str.
            Human-readable name for a layer (Dense_0, Conv2D_1, etc.). Used for debugging and printing summary of net.
        activation: str.
            Name of activation function to apply within the layer (e.g. 'relu', 'linear').
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        do_batch_norm. bool:
            Whether to do batch normalization in the layer.
            NOTE: Ignore Weeks 1, 2 and until instructed otherwise.
        do_batch_norm. float:
            The batch normalization momentum hyperparamter.
            NOTE: Ignore Weeks 1, 2 and until instructed otherwise.
        do_layer_norm. bool:
            Whether to do layer normalization in the layer.
            NOTE: Ignore until instructed otherwise later in the semester.

        TODO: Make instance variables for each of the constructor parameters
        '''
        self.layer_name = layer_name
        self.act_fun_name = activation
        self.prev_layer_or_block = prev_layer_or_block
        self.do_batch_norm = do_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.do_layer_norm = do_layer_norm

        self.wts = None
        self.b = None
        self.output_shape = None

        # We need to make this tf.Variable so this boolean gets added to the static graph when net compiled. Otherwise,
        # bool cannot be updated during training when using @tf.function
        self.is_training = tf.Variable(False, trainable=False)

        # The following relates to features you will implement later in the semester. Ignore for now.
        self.bn_gain = None
        self.bn_bias = None
        self.bn_mean = None
        self.bn_stdev = None
        self.ln_gain = None
        self.ln_bias = None



    def get_name(self):
        '''Returns the human-readable string name of the current layer.'''
        return self.layer_name

    def get_act_fun_name(self):
        '''Returns the activation function string name used in the current layer.'''
        return self.act_fun_name

    def get_prev_layer_or_block(self):
        '''Returns a reference to the Layer object that represents the layer below the current one.'''
        return self.prev_layer_or_block

    def get_wts(self):
        '''Returns the weights of the current layer'''
        return self.wts

    def get_b(self):
        '''Returns the bias of the current layer'''
        return self.b

    def has_wts(self):
        '''Does the current layer store weights? By default, we assume it does not (i.e. always return False).'''
        return self.wts is not None

    def get_mode(self):
        '''Returns whether the Layer is in a training state.

        HINT: Check out the instance variables above...
        '''
        return self.is_training

    def set_mode(self, is_training):
        '''Informs the layer whether the neural network is currently training. Used in Dropout and some other layer
        types.

        Parameters:
        -----------
        is_training: bool.
            True if the network is currently training, False otherwise.

        TODO: Update the appropriate instance variable according to the state passed into this method.
        NOTE: Notice the instance variable is of type tf.Variable. We do NOT want to use = assignment, otherwise
        TensorFlow will create a new node everything this method is called and the variable in the compiled network
        graph will NOT be updated. Practically, this means:
        1. You will pull your hair out wondering why the training state of the network is NOT being updated, even though
        you pass True into this method :(
        2. You will create a memory leak because TF will many duplicate nodes in the compiled graph (that do nothing).

        Use the `assign` method on the instance variable to update the training state.
        This method should be a one-liner.
        '''
        self.is_training.assign(is_training)


    def init_params(self, input_shape):
        '''Initializes the Layer's parameters (wts + bias), if it has any.

        Leave this parent method empty — subclasses should implement this.
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net_in on the input tensor `x`.

        Leave this parent method empty — subclasses should implement this.
        '''
        pass

    def compute_net_activation(self, net_in):
        '''Computes the appropriate activation based on the `net_in` values passed in.

        In Project 1, the following activation functions should be supported: 'relu', 'linear', 'softmax'.

        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, ...)
            The net input computed in the current layer.
            NOTE:

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, ...).
            The activation computed on the current mini-batch.

        NOTE:
        - `B` is the batch size.
        - The ... above in the net_in shape refers to the fact that the number of non-batch dimensions could be
        different, depending on the layer (e.g. Dense vs Conv2D). Do NOT write code that makes assumptions about which
        or how many non-batch dimensions are available.
        - To prevent silent bugs, I suggest throwing an error if the user sets an unsupported activation function.
        - Unless instructed otherwise, you may use the activation function implementations provided by the low level
        TensorFlow API here (You already implemented them in CS343 so you have earned it :)
        '''

        if self.act_fun_name == 'relu':
            return tf.nn.relu(net_in)
        if self.act_fun_name == 'linear':
            return net_in
        elif self.act_fun_name == 'softmax':
            return tf.nn.softmax(net_in)
        else:
            raise ValueError(f'Unknown activation function {self.act_fun_name}')

    def __call__(self, x):
        '''Do a forward pass thru the layer with mini-batch `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, ...).
            The activation computed on the current mini-batch.

        NOTE:
        - `B` is the batch size.
        - The ... above in the net_in shape refers to the fact that the number of non-batch dimensions could be
        different, depending on the layer (e.g. Dense vs Conv2D). Do NOT write code that makes assumptions about which
        or how many non-batch dimensions are available.

        TODO:
        1. Do the forward pass thru the layer (i.e. compute net_in and net_act).
        2. Before the method ends, check to see if `self.output_shape` is None. If it is, that means we are processing
        our very first mini-batch of data ever (e.g. at the beginning of training). If `self.output_shape` is None,
        set it to the shape of the layer's activation, represented as a Python list. You can convert something into a
        Python list by calling the `list` function — e.g. `list(blah)`.
        '''
        net_in = self.compute_net_input(x)
        net_act = self.compute_net_activation(net_in)

        if self.output_shape is None:
            self.output_shape = list(net_act.shape)
            
        return net_act

        

    def get_params(self):
        '''Gets a list of all the parameters learned by the layer (wts, bias, etc.).

        This method is provided to you, so you should not need to modify it.
        '''
        params = []

        if self.wts is not None:
            params.append(self.wts)
        if self.b is not None and self.b.trainable:
            params.append(self.b)
        # The following relates to features you will implement later in the semester. Running code should not
        # affect anything you are implementing now.
        if self.bn_gain is not None:
            params.append(self.bn_gain)
        if self.bn_bias is not None:
            params.append(self.bn_bias)
        if self.ln_gain is not None:
            params.append(self.ln_gain)
        if self.ln_bias is not None:
            params.append(self.ln_bias)

        return params

    def get_kaiming_gain(self):
        '''Returns the Kaiming gain that is appropriate for the current layer's activation function.

        (Week 2)

        Returns:
        --------
        float.
            The Kaiming gain.
        '''
        pass

    def is_doing_batchnorm(self):
        '''Returns whether the current layer is using batch normalization.

        (Week 3)

        Returns:
        --------
        bool.
            True if the layer has batch normalization turned on, False otherwise.
        '''
        pass

    def init_batchnorm_params(self):
        '''Initializes the trainable and non-trainable parameters used in batch normalization. This includes the
        batch norm gain and bias, as well as the moving average mean and standard deviation.

        (Week 3)

        NOTE: The following instructions have a lot of words but you ultimately will not need to write much code!

        TODO:
        1a. Determine out the shape of each set of batch norm parameters (they all have the same shape).
        The tricky thing is that different layers could have different shapes of their net_acts (e.g. (B, H) for Dense vs (B, Iy, Ix, K) for
        Conv2D). Ideally, we would like to NOT special case this computation, so that it will work for any type of
        layer. The trick is that each batch norm variable has the same shape of the number of units in the layer, which
        is the LAST axis of the layer activations. This is (H,) for Dense, (K,) for Conv2D.
        1b. To prevent bugs when using broadcasting when computing batch normalization, we should add
        singleton dimensions so the number of dimensions of the batch norm parameters matches that of the layer
        activations.
        Examples:
            - In a Dense layer the activations would have shape (B, H) so the batch norm parameters should
            have shape (1, H).
            - In a Conv2D layer the activations would have shape (B, Iy, Ix, K) so the batch norm parameters should
            have shape (1, 1, 1, H).
        1c. Create a Python list that represents the shape of the batch norm parameters (e.g. (1, H) or (1, 1, 1, K)).
        Be careful not to hard-code the number of ones.
        2. Initialize the batch norm gain and bias as tf.Variables.
        3. Initialize the batch norm moving avg mean and standard deviation as tf.Variables, but set the trainable
        keyword arument of tf.Variable to `False` since back prop is NOT updating these variables (so we do not need
        TF to track their gradients for us).
        4. Turn "off' the normal bias in the layer. Do this by setting the bias to the scalar Tf.Variable `0.0`. Make it
        not trainable.
        '''
        # KEEP ME
        if not self.do_batch_norm:
            return


    def compute_batch_norm(self, net_in, eps=0.001):
        '''Computes the batch normalization based on on the net input `net_in`.

        Leave this parent method empty — subclasses should implement this.
        '''

    def is_doing_layernorm(self):
        '''Check if layer normalization is enabled. True if layer normalization is enabled, False otherwise.

        (Ignore until later in the semester)
        '''
        pass

    def init_layernorm_params(self, x):
        '''Initializes the parameters for layer normalization if layer normalization is enabled.

        (Ignore until later in the semester)

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...).
            Input tensor to be normalized.

        TODO:
        1. Initialize the layer normalization gain and bias instance vars.
        2. Turn off the ordinary bias by replacing the bias with a non-trainable tf.Variable scalar of 0.
        '''
        if not self.do_layer_norm:
            return

    def compute_layer_norm(self, x, eps=0.001):
        '''Computes layer normalization for the input tensor. Layer normalization normalizes the activations of the
        neurons in a layer for each data point independently, rather than across the batch.

        (Ignore until later in the semester)

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M).
            Input tensor to be normalized.
        eps: float.
            A small constant added to the standard deviation to prevent division by zero. Default is 0.001.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, M).
            The normalized tensor with the same shape as the input tensor.
        '''
        pass

    def gelu(self, net_in):
        '''Applies the Gaussian Error Linear Unit (GELU) activation function.

        (Ignore until later in the semester)

        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, M)
            The net input to which the activation function should be applied.

        Returns:
        --------
        tf.constant. shape=(B, M)
            Output tensor after applying the GELU activation function.
        '''
        pass


class Dense(Layer):
    '''Neural network layer that uses Dense net input.'''
    def __init__(self, name, units, activation='relu', wt_scale=1e-3, prev_layer_or_block=None,
                 wt_init='normal', do_batch_norm=False, do_layer_norm=False):
        '''Dense layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Dense_0). Used for debugging and printing summary of net.
        units: int.
            Number of units in the layer (H).
        activation: str.
            Name of activation function to apply within the layer (e.g. 'relu', 'linear').
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore Week 1 and until instructed otherwise.
        do_batch_norm. bool:
            Whether to do batch normalization in the layer.
            NOTE: Ignore Weeks 1, 2 and until instructed otherwise.
        do_layer_norm. bool:
            Whether to do layer normalization in the layer.
            NOTE: Ignore until instructed otherwise later in the semester.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''
        super().__init__(name, activation, prev_layer_or_block,
                         do_batch_norm=do_batch_norm,
                         do_layer_norm=do_layer_norm)

    def has_wts(self):
        '''Returns whether the Dense layer has weights. This is always true so always return... :)'''
        pass

    def init_params(self, input_shape):
        '''Initializes the Dense layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list.
            The anticipated shape of mini-batches of input that the layer will process. For most of the semester,
            this list will look: (B, M).

        NOTE:
        - Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph during
        training.
        - For consistency with the test code, initialize your wts before your biases.
        - DO NOT assume the number of units in the previous layer is input_shape[1]. Instead, determine it as the last
        element of input_shape. This may sound silly, but doing this will prevent you from having to modify this method
        later in the semester :)
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net input for the current Dense layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, M).
            Input from the layer beneath in the network.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H).
            The net_in.

        NOTE: This layer uses lazy initialization. This means that if the wts are currently None when we enter this
        method, we should call `init_params` to initialize the parameters!
        '''
        pass

    def compute_batch_norm(self, net_in, eps=0.001):
        '''Computes the batch normalization in a manner that is appropriate for Dense layers.

        (Week 3)

        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, H).
            The net input computed on the current mini-batch.
        eps: float.
            A small "fudge factor" to prevent division by 0 when standardizing the net_in.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, H).
            The net_in, standardized according to the batch normalization algorithm.

        NOTE:
        - The moving avg parameters should only be updated when the network is in training mode.
        - The moving average parameters should be computed over the non-batch dimensions.
        - You should use the `assign` method to update the moving avg parameters (i.e. NO =). Even though they are
        not trainable, we still want to be able to update them after you have TensorFlow compile the neural network
        graph.
        - The net input should be normalized using the current mini-batch mean/stddev during training and the moving avg
        parameters when not training.
        '''
        pass

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Dense layer output({self.layer_name}) shape: {self.output_shape}'


class Dropout(Layer):
    '''A dropout layer that nixes/zeros out a proportion of the net input signals.'''
    def __init__(self, name, rate, prev_layer_or_block=None):
        '''Dropout layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        rate: float.
            Proportion (between 0.0 and 1.0.) of net_in signals to drop/nix within each mini-batch.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net input for the current Dropout layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...).
            Input from the layer beneath in the network. This could be 2D (e.g. (B, H)) if the preceding layer is Dense
            or another number of dimensions (e.g. 4D (B, Iy, Ix, K) for Conv2D).

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, ...), same shape as the input `x`.
            The net_in.

        NOTE:
        - Remember that computing the Dropout net_in operates differently in train and non-train modes.
        - Because the shape of x could be variable in terms of the number of dimensions (e.g. 2D, 4D), do not hard-code
        axes when working with shapes. For example, blah.shape[2] is considered hard coding because blah may not always
        have an axis 2.
        '''
        pass

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Dropout layer output({self.layer_name}) shape: {self.output_shape}'


class Flatten(Layer):
    '''A flatten layer that flattens the non-batch dimensions of the input signal.'''
    def __init__(self, name, prev_layer_or_block=None):
        '''Flatten layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net input for the current Flatten layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, ...).
            Input from the layer beneath in the network. Usually the input will come from Conv2D or MaxPool2D layers
            in which case the shape of `x` is 4D: (B, Iy, Ix, K).

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, F),
            The net_in. Here `F` is the number of units once the non-batch dimensions of the input signal `x` are
            flattened out.

        NOTE:
        - While the shape of the input `x` will usually be 4D, it is better to not hard-code this just in case.
        For example, do NOT do compute the number of non-batch inputs as x.shape[1]*x.shape[2]*x.shape[3]
        '''
        pass

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Flatten layer output({self.layer_name}) shape: {self.output_shape}'


class MaxPool2D(Layer):
    '''A 2D maxpooling layer.'''
    def __init__(self, name, pool_size=(2, 2), strides=1, prev_layer_or_block=None, padding='VALID'):
        '''MaxPool2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        pool_size. tuple. len(pool_size)=2.
            The horizontal and vertical size of the pooling window.
            These will always be the same. For example: (2, 2), (3, 3), etc.
        strides. int.
            The horizontal AND vertical stride of the max pooling operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        padding: str.
            Whether or not to pad the input signal before performing max-pooling in TensorFlow str format.
            Supported options: 'VALID', 'SAME'
            Most often, this will be 'VALID' for no padding, like we are used to.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net input for the current MaxPool2D layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. Should be 4D (e.g. from a Conv2D or MaxPool2D layer).
            K1 refers to the number of units/filters in the PREVIOUS layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            The net_in. K2 refers to the number of units/filters in the CURRENT layer.

        TODO: Compute the max pooling using TensorFlow's max_pool2d function. You can leave the data_format
        keyword arguments to its default value.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d
        '''
        pass

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'MaxPool2D layer output({self.layer_name}) shape: {self.output_shape}'


class Conv2D(Layer):
    '''A 2D convolutional layer'''
    def __init__(self, name, units, kernel_size=(1, 1), strides=1, activation='relu', wt_scale=1e-3,
                 prev_layer_or_block=None, wt_init='normal', do_batch_norm=False):
        '''Conv2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        units: ints.
            Number of convolutional filters/units (K).
        kernel_size: tuple. len(kernel_size)=2.
            The horizontal and vertical extent (pixels) of the convolutional filters.
            These will always be the same. For example: (2, 2), (3, 3), etc.
        strides. int.
            The horizontal AND vertical stride of the convolution operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        activation: str.
            Name of the activation function to apply in the layer.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore Week 1 and until instructed otherwise.
        do_batch_norm. bool:
            Whether to do batch normalization in this layer.
            NOTE: Ignore Weeks 1, 2 and until instructed otherwise.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''
        pass

    def has_wts(self):
        '''Returns whether the Conv2D layer has weights. This is always true so always return... :)'''
        pass

    def init_params(self, input_shape):
        '''Initializes the Conv2D layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list. len(input_shape)=4.
            The anticipated shape of mini-batches of input that the layer will process: (B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.

        NOTE:
        - Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph during
        training.
        - For consistency with the test code, initialize your wts before your biases.
        '''
        pass

    def compute_net_input(self, x):
        '''Computes the net input for the current Conv2D layer. Uses SAME boundary conditions.

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
        2. Compute the convolution using TensorFlow's conv2d function. You can leave the dilations and data_format
        keyword arguments to their default values / you do not need to specify these parameters.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

        NOTE: Don't forget the bias!
        '''
        pass

    def compute_batch_norm(self, net_in, eps=0.001):
        '''Computes the batch normalization in a manner that is appropriate for Conv2D layers.

        (Week 3)

        Parameters:
        -----------
        net_in: tf.constant. tf.float32s. shape=(B, Iy, Ix, K).
            The net input computed on the current mini-batch.
        eps: float.
            A small "fudge factor" to prevent division by 0 when standardizing the net_in.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K).
            The net_in, standardized according to the batch normalization algorithm.

        NOTE:
        - Your implementation will be nearly identical to your implementation in Dense so it probably want to copy-paste
        that as a starting point.
        - The difference is in the moving average parameters: as in Dense, they are computed over the non-batch
        dimensions. This likely requires a small code change.
        '''
        pass

    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Conv2D layer output({self.layer_name}) shape: {self.output_shape}'
