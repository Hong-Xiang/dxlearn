from dxl.learn.model import Conv2D, Inception, Dense
import tensorflow as tf
import numpy as np
from doufo.tensor import shape


def test_conv2d_basic(tensorflow_test):
    m = Conv2D('conv1', 64, 3)
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    y = m(x)
    assert shape(y) == [32, 64, 64, 64]


def test_conv2d_get_params():
    m = Conv2D('conv2', 64, 3)
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    y = m(x)
    assert m.parameters[0].get_shape() == (3, 3, 3, 64)
    assert m.parameters[1].get_shape() == (64,)


def test_Inception_basic():
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    m1 = Conv2D('conv3', 64, 3)
    m2 = Dense('dense1', 128)
    paths = []
    for i in range(3):
        paths.append(Conv2D('conv' + str(i), 32 + 32 * i, 3))
    m = Inception('inception1', m1, m2, paths)
    m.set_merger_axis(3)
    y = m(x)
    assert shape(y) == [32, 64, 64, 128]


def test_Inception_parameter():
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    m1 = Conv2D('conv4', 64, 3)
    m2 = Dense('dense2', 128)
    paths = []
    for i in range(3):
        paths.append(Conv2D('conv' + str(i), 32 + 32 * i, 3))
    m = Inception('inception2', m1, m2, paths)
    m.set_merger_axis(3)
    y = m(x)
    assert m.parameters[0].get_shape() == (3, 3, 3, 64)
    assert m.parameters[1].get_shape() == (64,)
    assert m.parameters[2].get_shape() == (3, 3, 64, 32)
    assert m.parameters[3].get_shape() == (32,)
