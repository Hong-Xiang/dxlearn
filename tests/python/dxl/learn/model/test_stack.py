import tensorflow as tf
import numpy as np
from dxl.learn.model.stack import Stack, StackedConv2D
from dxl.learn.model.stack import StackedResidualConv, StackedResidualIncept
from dxl.learn.test import TestCase, UnitBlock


class TestStack(TestCase):
    def get_input(self):
        return np.ones([1, 100, 100, 3], dtype="float32")

    def make_model(self):
        return UnitBlock("unitblock_test")

    def test_Stack(self):
        x = self.get_input()
        nb_layer = 2
        y_ = x

        stack_ins = Stack(
            'Stack_test', tf.constant(x), self.make_model(), nb_layer)
        y = stack_ins()
        with self.variables_initialized_test_session() as sess:
            y = sess.run(y)
            self.assertAllEqual(y, y_)


class TestSpecialStack(TestCase):
    def get_input(self):
        return np.ones([1, 100, 100, 3], dtype="float32")

    def make_model(self):
        return UnitBlock("unitblock_test")

    def test_StackedResidualIncept(self):
        x = self.get_input()
        nb_layers = 2
        y_ = x
        
        stackedResidualincept_ins = StackedResidualIncept(
            'StackedResidualIncept_test',
            inputs=tf.constant(x),
            nb_layers=nb_layers,
            graph=self.get_model())
        y = stackedResidualincept_ins()
        with self.variables_initialized_test_session() as sess:
            y = sess.run(y)
            self.assertAllEqual(y, y_)

    def test_StackedResidualConv(self):
        x = self.get_input()
        nb_layers = 2
        y_ = x

        stackedresidualconv_ins = StackedResidualConv(
            'StackedResidualConv_test',
            inputs=tf.constant(x),
            nb_layers=nb_layers,
            graph=self.get_model())
        y = stackedresidualconv_ins()
        with self.variables_initialized_test_session() as sess:
            y = sess.run(y)
            self.assertAllEqual(y, y_)


class TestSpecialStackDefaultBlock(TestCase):
    def get_input(self):
        return np.ones([1, 10, 10, 3], dtype="float32")

     def test_StackedResidualInceptDef(self):
        x = self.get_input()
        nb_layers = 2
        # default ResidualIncept ratio=0.3
        stackedResidualincept_ins = StackedResidualIncept(
            'StackedResidualInceptDef_test',
            inputs=tf.constant(x),
            nb_layers=nb_layers)
        y = stackedResidualincept_ins()
        self.assertAllEqual(y.shape, (1, 10, 10, 3))

    def test_StackedResidualConvDef(self):
        x = self.get_input()
        nb_layers = 2
        # default ResidualIncept ratio=0.1
        stackedresidualconv_ins = StackedResidualConv(
            'StackedResidualConvDef_test',
            inputs=tf.constant(x),
            nb_layers=nb_layers)
        y = stackedresidualconv_ins()
        self.assertAllEqual(y.shape, (1, 10, 10, 3))

        