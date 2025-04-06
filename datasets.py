'''datasets.py
Loads and preprocesses datasets for use in neural networks.
Trey Tuscai and Gordon Doore
CS444: Deep Learning
'''
import tensorflow as tf
import numpy as np


def load_dataset(name):
    '''Uses TensorFlow Keras to load and return  the dataset with string nickname `name`.

    Parameters:
    -----------
    name: str.
        Name of the dataset that should be loaded. Support options in Project 1: 'cifar10', 'mnist'.

    Returns:
    --------
    x: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        The training set (preliminary).
    y: tf.constant. tf.int32s.
        The training set int-coded labels (preliminary).
    x_test: tf.constant. tf.float32s.
        The test set.
    y_test: tf.constant. tf.int32s.
        The test set int-coded labels.
    classnames: Python list. strs. len(classnames)=num_unique_classes.
        The human-readable string names of the classes in the dataset. If there are 10 classes, len(classnames)=10.

    Summary of preprocessing steps:
    -------------------------------
    1. Uses tf.keras.datasets to load the specified dataset training set and test set.
    2. Loads the class names from the .txt file downloaded from the project website with the same name as the dataset
        (e.g. cifar10.txt).
    3. Features: Converted from UINT8 to tf.float32 and normalized so that a 255 pixel value gets mapped to 1.0 and a
        0 pixel value gets mapped to 0.0.
    4. Labels: Converted to tf.int32 and flattened into a tensor of shape (N,).

    Helpful links:
    https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
    https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
    '''
    if name.lower() == 'cifar10':
        (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        classnames_file = 'data/cifar10.txt'
    elif name.lower() == 'mnist':
        (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x = np.expand_dims(x, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        classnames_file = 'data/mnist.txt'
    elif name.lower() == 'cifar100':
        (x, y), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        classnames_file = 'data/cifar100.txt'
    else:
        raise ValueError("Supported datasets: 'cifar10', 'mnist'.")

    x, x_test = x.astype('float32') / 255.0, x_test.astype('float32') / 255.0

    y, y_test = tf.convert_to_tensor(
        y, dtype=tf.int32), tf.convert_to_tensor(y_test, dtype=tf.int32)
    y, y_test = tf.reshape(y, [-1]), tf.reshape(y_test, [-1])

    try:
        with open(classnames_file, 'r') as f:
            classnames = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Class names file '{classnames_file}' not found.")

    return x, y, x_test, y_test, classnames


def standardize(x_train, x_test, eps=1e-10):
    '''Standardizes the image features using the global RGB triplet method.

    Parameters:
    -----------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features (preliminary).
    x_test: tf.constant. tf.float32s. shape=(N_test, I_y, I_x, n_chans).
        Test set features.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Standardized training set features (preliminary).
    tf.constant. tf.float32s. shape=(N_test, I_y, I_x, n_chans).
        Standardized test set features (preliminary).
    '''
    mean = tf.reduce_mean(x_train, (0, 1, 2), keepdims=True)
    std = tf.math.reduce_std(x_train, (0, 1, 2), keepdims=True)

    x_train_standardized = (x_train - mean) / (std + eps)
    x_test_standardized = (x_test - mean) / (std + eps)

    return x_train_standardized, x_test_standardized


def train_val_split(x_train, y_train, val_prop=0.1):
    '''Subdivides the preliminary training set into disjoint/non-overlapping training set and validation sets.
    The val set is taken from the end of the preliminary training set.

    Parameters:
    -----------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features (preliminary).
    y_train: tf.constant. tf.int32s. shape=(N_train,).
        Training set class labels (preliminary).
    val_prop: float.
        The proportion of preliminary training samples to reserve for the validation set. If the proportion does not
        evenly subdivide the initial N, the number of validation set samples should be rounded to the nearest int.

    Returns:
    --------
    tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        Training set features.
    tf.constant. tf.int32s. shape=(N_train,).
        Training set labels.
    tf.constant. tf.float32s. shape=(N_val, I_y, I_x, n_chans).
        Validation set features.
    tf.constant. tf.int32s. shape=(N_val,).
        Validation set labels.
    '''
    num_samps = tf.shape(x_train)[0]
    num_val_samps = tf.cast(
        tf.round(tf.cast(num_samps, dtype=tf.float32) * val_prop), dtype=tf.int32)

    x_val = x_train[-num_val_samps:]
    y_val = y_train[-num_val_samps:]

    x_train_new = x_train[:-num_val_samps]
    y_train_new = y_train[:-num_val_samps]

    return x_train_new, y_train_new, x_val, y_val


def get_dataset(name, standardize_ds=True, val_prop=0.1):
    '''Automates the process of loading the requested dataset `name`, standardizing it (optional), and create the val
    set.

    Parameters:
    -----------
    name: str.
        Name of the dataset that should be loaded. Support options in Project 1: 'cifar10', 'mnist'.
    standardize_ds: bool.
        Should we standardize the dataset?
    val_prop: float.
        The proportion of preliminary training samples to reserve for the validation set. If the proportion does not
        evenly subdivide the initial N, the number of validation set samples should be rounded to the nearest int.

    Returns:
    --------
    x_train: tf.constant. tf.float32s. shape=(N_train, I_y, I_x, n_chans).
        The training set.
    y_train: tf.constant. tf.int32s.
        The training set int-coded labels.
    x_val: tf.constant. tf.float32s. shape=(N_val, I_y, I_x, n_chans).
        Validation set features.
    y_val: tf.constant. tf.int32s. shape=(N_val,).
        Validation set labels.
    x_test: tf.constant. tf.float32s.
        The test set.
    y_test: tf.constant. tf.int32s.
        The test set int-coded labels.
    classnames: Python list. strs. len(classnames)=num_unique_classes.
        The human-readable string names of the classes in the dataset. If there are 10 classes, len(classnames)=10.
    '''
    x_train, y_train, x_test, y_test, classnames = load_dataset(name)
    if standardize_ds:
        x_train, x_test = standardize(x_train, x_test)
    x_train, y_train, x_val, y_val = train_val_split(
        x_train, y_train, val_prop)
    return x_train, y_train, x_val, y_val, x_test, y_test, classnames
