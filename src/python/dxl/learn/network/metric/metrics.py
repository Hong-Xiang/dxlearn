import tensorflow as tf


def l2norm(data):
    with tf.name_scope('l2_norm'):
        return tf.sqrt(tf.reduce_mean(tf.square(data)))


def l1norm(data):
    with tf.name_scope('l1_norm'):
        return tf.reduce_mean(tf.abs(data))


def mean_square_error(label, data):
    with tf.name_scope('mse'):
        return l2norm(label - data)


def mse(label, data):
    return mean_square_error(label, data)


def rmse(label, data):
    with tf.name_scope('rmse'):
        return mse(label, data) / l2norm(label)


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
