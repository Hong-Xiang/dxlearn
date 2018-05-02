import tensorflow as tf


def mean_square_error(label, data):
    with tf.name_scope('mean_squared_error'):
        return tf.sqrt(tf.reduce_mean(tf.square(label - data)))


def rmse(label, data):
    with tf.name_scope('rmse'):
        rmse = tf.sqrt(tf.reduce_mean(tf.square(label - data)))
    return rmse


def psnr(label, data):
    with tf.name_scope('psnr'):
        r = rmse(label, data)
        maxv = tf.reduce_max(label)
        minv = tf.reduce_min(label)
        sca = 255.0 / (maxv - minv)
        ln = (label - minv) * sca
        tn = (data - minv) * sca
        rmv = rmse(ln, tn)
        value = 10.0 * tf.log((255.0**2.0) / (rmv**2.0)) / tf.log(10.0)
    return value
