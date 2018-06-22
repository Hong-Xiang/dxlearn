import tensorflow as tf

def mean_square_error(label, data):
    with tf.name_scope('mse'):
        mse, update_op = tf.metrics.mean_squared_error(label, data)
        return mse