from dxl.learn.model import Conv2D, Dense, Stack, Residual, Merge, Inception
import tensorflow as tf
from doufo.tensor import shape


def test_base_model_inheritance():
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    m1 = Conv2D('conv1', 32, 3)
    m2 = Dense('dense1', 160)
    m3 = Stack([m1, m2])
    res1 = m3(x)
    m4 = Conv2D('conv2', 160, 3)
    m5 = Residual('resi1', m4)
    res2 = m5(res1)
    m7 = Conv2D('conv3', 3, 3)
    m8 = Dense('dense2', 128)
    m6 = Inception('inc1', m7, m7, [m1, m8])
    m6.set_merger_axis(3)
    res3 = m6(res2)
    assert shape(res1) == [32, 64, 64, 160]
    assert shape(res2) == [32, 64, 64, 160]
    assert shape(res3) == [32, 64, 64, 3]
