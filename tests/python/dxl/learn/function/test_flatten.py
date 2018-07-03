from dxl.learn.test import TestCase
from dxl.learn.function import Flatten, flatten
import numpy as np
import tensorflow as tf
import cntk


def _assert_correct(x):
    np.testing.assert_array_equal(
        x, np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8]], dtype=np.float32))


def _input_tensor():
    return np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)


class TestFlatten(TestCase):
    def assertCorrent(self, x):
        _assert_correct(x)

    def test_np(self):
        x = flatten(_input_tensor())
        self.assertCorrent(x)

    def test_tf(self):
        with self.graph_on_cpu() as g:
            x = tf.constant(_input_tensor())
            y = flatten(x)
            with self.test_session() as sess:
                result = sess.run(y)
        self.assertCorrent(result)


def test_cntk():
    x = cntk.input([2, 2], np.float32)
    y = flatten(x)
    result = y.eval({x: _input_tensor()})
    _assert_correct(result)
