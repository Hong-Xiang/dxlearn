from dxl.learn.model import Conv2D, Inception,Dense
import tensorflow as tf
import numpy as np
from doufo.tensor import shape


def test_conv2d_basic():
    m = Conv2D(64, 3, name='conv')
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    y = m(x)
    assert shape(y) == [32, 64, 64, 64]

def test_conv2d_get_params():
    m = Conv2D(64, 3, name='conv')
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    y = m(x)
    assert m.parameters[0].get_shape() == (3,3,3,64)
    assert m.parameters[1].get_shape() == (64,)

def test_Inception_basic():
    x = tf.constant(np.ones([32,64,64,3],np.float32))
    m1 = Conv2D(64,3,name='conv')
    m2 = Dense(128)
    paths = []
    for i in range(3):
        paths[i] = Conv2D(32+32*i,3,name='conv'+str(i))

    m = Inception('inception1',m1,paths,m2)
    y = m(x)
