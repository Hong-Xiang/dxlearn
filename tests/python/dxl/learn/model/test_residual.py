from dxl.learn.model.residual import Residual
from doufo import identity
from dxl.learn.config import config_with_name, clear_config
from dxl.learn.model import Conv2D, parameters
import tensorflow as tf
from doufo.tensor import shape, all_close


def test_residual_config_direct(clean_config):
    r = Residual('res', identity, 0.1)
    assert r.config[Residual.KEYS.CONFIG.RATIO] == 0.1


def test_residual_config_proxy(clean_config):
    c = config_with_name('res')
    c[Residual.KEYS.CONFIG.RATIO] = 0.1
    r = Residual('res', identity)
    assert r.config[Residual.KEYS.CONFIG.RATIO] == 0.1


def test_residual_config_default(clean_config):
    r = Residual('res', identity)
    assert r.config[Residual.KEYS.CONFIG.RATIO] == 0.3


def test_residual_config_proxy_direct_conflict(clean_config):
    c = config_with_name('res')
    c[Residual.KEYS.CONFIG.RATIO] = 0.1
    r = Residual('res', identity, 0.2)
    assert r.config[Residual.KEYS.CONFIG.RATIO] == 0.2


def test_residual_basic(clean_config):
    m1 = Conv2D('conv1', 3, 3)
    r = Residual('res1', m1)
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    res1 = r(x)
    res2 = m1(x)
    assert shape(res1) == [32, 64, 64, 3]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a, b = sess.run([res1, res2])
        assert all_close(a, b) is False


def test_residual_parameters(clean_config):
    m1 = Conv2D('conv1', 3, 3)
    r = Residual('res1', m1)
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    res1 = r(x)
    assert r.parameters[0].get_shape() == (3, 3, 3, 3)
