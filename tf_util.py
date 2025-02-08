'''tf_util.py
A few helper functions provided to you for various projects in CS444
Oliver W. Layton
CS444: Deep Learning
'''
import tensorflow as tf


def arange_index(x, y):
    '''Reproduces arange indexing from NumPy in TensorFlow. I.e. Pick out the values in the column
    indices `y` as you go down the rows.

    Parameters:
    -----------
    x: tf.constant. tf.float32s. shape=(B, C).
        A 2D tensor that we want to index with arange indexing.
    y: tf.constant. tf.float32s. shape=(B,).
        The column indices to pick out from each row of `x`

    Returns:
    --------
    tf.constant. tf.float32s. shape=(B,).
        Values from `x` extract from columns specified by `y`.

    Example:
    --------
    x = [[1., 2., 3.],
         [4., 5., 6.],
         [7., 8., 9.],
         [3., 2., 1.]]
    y = [1, 0, 2, 1]
    returns: [2., 4., 9., 2.]
    '''
    rows = tf.range(len(x))
    rc_tuples = tf.stack([rows, y], axis=1)
    return tf.gather_nd(x, rc_tuples)

def interleave_cols(x, y):
    # Combine tensors x and y into one larger tensor column-wise (i.e. we glue columns together)
    stacked = tf.stack([x, y], axis=1)
    # Interleave columns across the tensors
    interleaved = tf.transpose(stacked, (0, 2, 1))
    # Now smush out the two segregation between x and y so that we lose the distinction.
    smushed = tf.reshape(interleaved, (len(x), -1))
    return smushed

def tril(sz, num_leading_singleton_dims=2):
    shape = num_leading_singleton_dims*[1]
    shape.extend([sz, sz])
    return tf.reshape(tf.linalg.band_part(tf.ones([sz, sz]), num_lower=-1, num_upper=0), shape)
