import tensorflow as tf

def mean_square_error(label, data, weights=1.0):
    with tf.name_scope('mse'):
        return tf.losses.mean_squared_error(label, data, weights)

def softmax_cross_entropy(label, data, weights=1.0):
    with tf.name_scope('sce'):
        return tf.losses.softmax_cross_entropy(label, data, weights)


# Aliases.
mse = MSE = mean_square_error
sce = SCE = softmax_cross_entropy