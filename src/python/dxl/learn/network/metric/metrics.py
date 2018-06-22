import tensorflow as tf

def mean_square_error(label, data):
    with tf.name_scope('mse'):
        return tf.metrics.mean_squared_error(label, data)