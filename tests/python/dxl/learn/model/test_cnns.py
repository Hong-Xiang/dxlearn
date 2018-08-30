from dxl.learn.model import Conv2D
import tensorflow as tf
import numpy as np
from doufo.tensor import shape


def test_conv2d_basic(tensorflow_test):
    m = Conv2D(64, 3, name='conv')
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    y = m(x)
    assert shape(y) == [32, 64, 64, 64]
