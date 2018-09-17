from dxl.learn.function.activation import *
import tensorflow as tf
from doufo.tensor import Tensor
import math
from doufo.tensor.binary import all_close


def test_relu_with_tf_tensor():
    x = tf.constant([-1, -2, -3, 1, 2, 3], dtype=tf.float32)
    res = relu(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert all_close(res, [0, 0, 0, 1, 2, 3]) is True


def test_relu_with_tensor():
    x = tf.constant([-1, -2, -3, 1, 2, 3], dtype=tf.float32)
    x = Tensor(x)
    res = relu(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert all_close(res, [0, 0, 0, 1, 2, 3]) is True


def test_swish_with_tf_tensor():
    x = tf.constant([-1, -2, 0, 1, 2], dtype=tf.float32)
    res = swish(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert res[2] == 0
    assert (res[0] - res[1]) < (res[4] - res[3])


def test_swish_with_tensor():
    x = tf.constant([-1, -2, 0, 1, 2], dtype=tf.float32)
    x = Tensor(x)
    res = swish(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert res[2] == 0
    assert (res[0] - res[1]) < (res[4] - res[3])


def test_elu_with_tf_tensor():
    x = tf.constant([-1, 0, 1], dtype=tf.float32)
    res = elu(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert abs(res[0] - (-0.63212055)) < math.pow(10, -8)
    assert res[1] == 0
    assert res[2] == 1


def test_elu_with_tensor():
    x = tf.constant([-1, 0, 1], dtype=tf.float32)
    x = Tensor(x)
    res = elu(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert abs(res[0] - (-0.63212055)) < math.pow(10, -8)
    assert res[1] == 0
    assert res[2] == 1


def test_celu_with_tf_tensor():
    x = tf.constant([-1, 0, 1], dtype=tf.float32)
    res = celu(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert abs(res[0] - (-0.63212055)) < math.pow(10, -8)
    assert res[1] == 0
    assert abs(res[-1] - (-0.63212055)) < math.pow(10, -8)
    assert res[3] == 1
    assert res[2] == 1
    assert res[4] == 0


def test_celu_with_tensor():
    x = tf.constant([-1, 0, 1], dtype=tf.float32)
    x = Tensor(x)
    res = celu(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert abs(res[0] - (-0.63212055)) < math.pow(10, -8)
    assert res[1] == 0
    assert abs(res[-1] - (-0.63212055)) < math.pow(10, -8)
    assert res[3] == 1
    assert res[2] == 1
    assert res[4] == 0


def test_selu_with_tf_tensor():
    x = tf.constant([-1, 0, 1], dtype=tf.float32)
    res = selu(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
    assert abs(res[0] - (-1.1113307)) < math.pow(10, -7)
    assert abs(res[2] - 1.050701) < math.pow(10, -6)
    assert res[1] == 0


def test_selu_with_tensor():
    x = tf.constant([-1, 0, 1], dtype=tf.float32)
    x = Tensor(x)
    res = selu(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res.unbox())
    assert abs(res[0] - (-1.1113307)) < math.pow(10, -7)
    assert abs(res[2] - 1.050701) < math.pow(10, -6)
    assert res[1] == 0
