from dxl.learn.function.losses import *
import tensorflow as tf
from doufo.tensor import Tensor
import math


def test_mean_square_error_with_tf_tensor():
    x = tf.constant([2, 2, 2], dtype=tf.float32)
    y = tf.constant([1, 1, 1], dtype=tf.float32)
    res = mean_square_error(y, x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert res == 1


def test_mean_square_error_with_tensor():
    x = tf.constant([2, 2, 2], dtype=tf.float32)
    y = tf.constant([1, 1, 1], dtype=tf.float32)
    x = Tensor(x)
    y = Tensor(y)
    res = mean_square_error(y, x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert res == 1


def test_absolute_error_with_tf_tensor():
    x = tf.constant([-2, -2, -2], dtype=tf.float32)
    y = tf.constant([1, 1, 1], dtype=tf.float32)
    res = absolute_error(y, x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert res == 3


def test_absolute_error_with_tensor():
    x = tf.constant([-2, -2, -2], dtype=tf.float32)
    y = tf.constant([1, 1, 1], dtype=tf.float32)
    x = Tensor(x)
    y = Tensor(y)
    res = absolute_error(y, x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert res == 3


def test_poisson_loss_with_tf_tensor():
    x = tf.constant([math.e, math.e, math.e], dtype=tf.float32)
    y = tf.constant([1, 1, 1], dtype=tf.float32)
    res = poisson_loss(y, x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert abs((math.e - res) - 1) < math.pow(10, -7)


def test_poisson_loss_with_tensor():
    x = tf.constant([math.e, math.e, math.e], dtype=tf.float32)
    y = tf.constant([1, 1, 1], dtype=tf.float32)
    x = Tensor(x)
    y = Tensor(y)
    res = poisson_loss(y, x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert abs((math.e - res) - 1) < math.pow(10, -7)


def test_log_poisson_loss_with_tf_tensor():
    x = tf.constant([math.e], dtype=tf.float32)
    y = tf.constant([1], dtype=tf.float32)
    res = log_poisson_loss(x, y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert res == 0


def test_log_poisson_loss_with_tensor():
    x = tf.constant([math.e], dtype=tf.float32)
    y = tf.constant([1], dtype=tf.float32)
    x = Tensor(x)
    y = Tensor(y)
    res = log_poisson_loss(x, y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert res == 0


def test_composite_loss_with_tf_tensor():
    x = tf.constant([2], dtype=tf.float32)
    y = tf.constant([1], dtype=tf.float32)
    loss = {mean_square_error: 1, absolute_error: 2}
    res = composite_loss(y, x, loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert res == 3


def test_composite_loss_with_tensor():
    x = tf.constant([2], dtype=tf.float32)
    y = tf.constant([1], dtype=tf.float32)
    x = Tensor(x)
    y = Tensor(y)
    loss = {mean_square_error: 1, absolute_error: 2}
    res = composite_loss(y, x, loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert res == 3
