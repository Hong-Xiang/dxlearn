from dxl.learn.model import Conv2D, Inception, Dense, DownSampling2D, UpSampling2D
import tensorflow as tf
import numpy as np
from doufo.tensor import shape
from dxl.core.config import *


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


def test_Inception_basic(clean_config):
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    m1 = Conv2D('conv3', 64, 3)
    m2 = Dense('dense1', 128)
    paths = [Conv2D('conv_{}'.format(i), 32 + 32 * i, 3) for i in range(3)]
    m = Inception('inception1', m1, m2, paths)
    y = m(x)
    assert shape(y) == [32, 64, 64, 128]


def test_Inception_parameter(clean_config):
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    m1 = Conv2D('conv4', 64, 3)
    m2 = Dense('dense2', 128)
    paths = [Conv2D('conv_{}'.format(i), 32 + 32 * i, 3) for i in range(3)]
    m = Inception('inception2', m1, m2, paths)
    y = m(x)
    assert m.parameters[0].get_shape() == (3, 3, 3, 64)
    assert m.parameters[1].get_shape() == (64,)
    assert m.parameters[2].get_shape() == (3, 3, 64, 32)
    assert m.parameters[3].get_shape() == (32,)


def test_downsampling_config(clean_config):
    m1 = DownSampling2D('dsp1', (3, 3), (1, 1), padding='same', method='max')
    assert m1.config['padding'] == 'same'
    assert m1.config['stride'] == (1, 1)
    assert m1.config['method'] == 'max'
    assert m1.config['pool_size'] == (3, 3)


def test_downsampling_config_default(clean_config):
    m1 = DownSampling2D('dsp2')
    assert m1.config['padding'] == 'valid'
    assert m1.config['stride'] == (1, 1)
    assert m1.config['method'] == 'mean'
    assert m1.config['pool_size'] == (1, 1)


def test_downsampling_basic(clean_config):
    m1 = DownSampling2D('dsp1', (3, 3), (1, 1), 'valid')
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    res = m1(x)
    assert shape(res) == [32, 62, 62, 3]


def test_upsampling_basic(clean_config):
    m1 = UpSampling2D('ups1', (2, 2), method=2)
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    res = m1(x)
    assert shape(res) == [32, 128, 128, 3]
