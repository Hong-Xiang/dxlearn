from dxl.learn.test import TestCase
from dxl.learn.function import OneHot
import numpy as np
import tensorflow as tf
import cntk


def _assert_correct(x):
    np.testing.assert_array_equal(
        x, np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0]]))


class TestOneHot(TestCase):
    def assertCorrent(self, x):
        _assert_correct(x)

    def test_np(self):
        x = OneHot(4)(np.array([0, 1, 2]))
        self.assertCorrent(x)

    def test_tf(self):
        f = OneHot(4)
        with self.graph_on_cpu() as g:
            x = tf.constant([0, 1, 2])
            y = f(x)
            with self.test_session() as sess:
                result = sess.run(y)
        self.assertCorrent(result)


def test_cntk():
    x = cntk.input([], np.float64)
    y = OneHot(4)(x)
    result = y.eval({x: np.array([0, 1, 2], np.float64)})
    _assert_correct(result)
