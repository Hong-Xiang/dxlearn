from dxl.learn.model import Conv2D, Dense, Stack, Residual, Merge, Inception
import tensorflow as tf
from doufo.tensor import shape


def test_base_model_inheritance_ver1(clean_config):
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    m1 = Conv2D('conv', 32, 3)
    m2 = Dense('dense', 160)
    m3 = Stack([m1, m2])
    res1 = m3(x)
    assert shape(res1) == [32, 64, 64, 160]


def test_base_model_inheritance_ver2(clean_config):
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    m4 = Conv2D('conv', 160, 3)
    m9 = Dense('dense', 160)
    x = m9(x)
    m5 = Residual('res', m4)
    res2 = m5(x)
    assert shape(res2) == [32, 64, 64, 160]


def test_base_model_inheritance_ver3(clean_config):
    x = tf.ones([32, 64, 64, 160], dtype=tf.float32)
    m1 = Conv2D('conv1', 32, 3)
    m7 = Conv2D('conv2', 3, 3)
    m8 = Dense('dense', 128)
    m6 = Inception('inception', m7, m7, [m1, m8])
    res3 = m6(x)
    assert shape(res3) == [32, 64, 64, 3]
