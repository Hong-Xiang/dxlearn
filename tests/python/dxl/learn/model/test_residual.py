import tensorflow as tf
import numpy as np
from dxl.learn.model.residual import Residual
from dxl.learn.test import TestCase, UnitBlock
from dxl.learn.model.residual import ResidualIncept, ResidualStackedConv

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


class TestSpecialResidual(TestCase):
    def get_input(self):
        return np.ones([1, 100, 100, 3], dtype="float32")

    def make_model(self):
        return UnitBlock("unitblock_test")

    def test_ResidualIncept(self):
        x = self.get_input()
        ratio = 0.5
        y_ = x + ratio * x

        residualincept_ins = ResidualIncept(
            'ResidualIncept_test',
            tf.constant(x),
            ratio,
            self.get_model())
        y = residualincept_ins()
        with self.variables_initialized_test_session() as sess:
            y = sess.run(y)
            self.assertAllEqual(y, y_)

    def test_ResidualStackedConv(self):
        x = self.get_input()
        ratio = 0.5
        y_ = x + ratio * x

        residualstackedconv_ins = ResidualStackedConv(
            'ResidualStackedConv_test',
            tf.constant(x),
            ratio,
            self.get_model())
        y = residualstackedconv_ins()
        with self.variables_initialized_test_session() as sess:
            y = sess.run(y)
            self.assertAllEqual(y, y_)


class TestSpecialResidualDefaultBlock(TestCase):
    def get_input(self):
        return np.ones([1, 10, 10, 3], dtype="float32")

    def test_ResidualInceptDef(self):
        x = self.get_input()
        ratio = 0.5
        residualincept_ins = ResidualIncept(
            'ResidualInceptDef_test', tf.constant(x), ratio)
        y = residualincept_ins()
        self.assertAllEqual(y.shape, (1, 10, 10, 3))

    def test_ResidualStackedConvDef(self):
        x = self.get_input()
        ratio = 0.5
        residualstackedconv_ins = ResidualStackedConv(
            'ResidualStackedConvDef_test',
            tf.constant(x),
            ratio)
        y = residualstackedconv_ins()
        self.assertAllEqual(y.shape, (1, 10, 10, 3))

    