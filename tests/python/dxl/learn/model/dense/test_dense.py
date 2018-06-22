import pytest
import tensorflow as tf
import numpy as np
from dxl.learn.model.dense import Dense
from dxl.learn.test import TestCase


class TestDense(TestCase):
    def get_input(self):
        return tf.constant(np.ones([5, 10]), tf.float32)

    def make_model(self):
        return Dense('dense', self.get_input(), 32)

    def test_result(self):
        dense = self.make_model()
        result = dense.make()
        except_shape = (5, 32)
        self.assertAllEqual(result.shape.as_list(), except_shape)