import tensorflow as tf
import numpy as np
from dxl.learn.model.stack import Stack
from dxl.learn.model.cnns import Conv2D
from doufo.tensor import shape


def test_stack_basic(clean_config):
    models = [Conv2D('conv1', 64, 3),
              Conv2D('conv2', 128, 3),
              Conv2D('conv3', 256, 3)]
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    st = Stack(models)
    res = st(x)
    assert shape(res) == [32, 64, 64, 256]


def test_stack_parameters(clean_config):
    models = [Conv2D('conv1', 64, 3),
              Conv2D('conv2', 128, 3),
              Conv2D('conv3', 256, 3)]
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    st = Stack(models)
    res = st(x)
    assert st.parameters[0].get_shape() == (3, 3, 3, 64)
