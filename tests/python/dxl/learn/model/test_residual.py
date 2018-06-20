import tensorflow as tf
import numpy as np
from dxl.learn.model.residual import Residual
from dxl.learn.test import TestCase, UnitBlock
from dxl.learn.model.residual import ResidualIncept

class TestResidual(TestCase):
    def get_input(self):
        return np.ones([1, 100, 100, 3], dtype="float32")

    def make_model(self):
        return UnitBlock("unitblock_test")

    def test_Residual(self):
        x = self.get_input()
        ratio = 0.5
        y_ = x + ratio * x

        residual_ins = Residual(
            'Residual_test', tf.constant(x), self.make_model(), ratio)
        y = residual_ins()
        with self.variables_initialized_test_session() as sess:
            y = sess.run(y)
            self.assertAllEqual(y, y_)