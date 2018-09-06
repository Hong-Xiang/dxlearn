from dxl.learn.model import Merge, Conv2D, Dense
import tensorflow as tf
from doufo.tensor import shape


def test_merge_basic(clean_config):
    m1 = Dense('d1', 256)
    m2 = Conv2D('conv1', 64, 3)
    m3 = Conv2D('conv2', 128, 3)
    m = Merge('merger1', merger=m1, models=[m2, m3], axis=3)
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    res = m(x)
    assert shape(res) == [32, 64, 64, 256]


def test_config_add_model(clean_config):
    m = Merge('merger1', axis=3)
    m1 = Dense('d1', 256)
    m2 = Conv2D('conv1', 64, 3)
    m3 = Conv2D('conv2', 128, 3)
    m.config_models([m2, m3])
    m.config_merger(m1)
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    res = m(x)
    assert shape(res) == [32, 64, 64, 256]


def test_merge_parameter(clean_config):
    m = Merge('merger1', axis=3)
    m1 = Dense('d1', 256)
    m2 = Conv2D('conv1', 64, 3)
    m3 = Conv2D('conv2', 128, 3)
    m.config_models([m2, m3])
    m.config_merger(m1)
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    res = m(x)
    assert m.parameters[0].get_shape() == (3, 3, 3, 64)
