from dxl.learn.test import TestCase
from dxl.learn.core import Model, Constant
import tensorflow as tf
import numpy as np


class TestModel(TestCase):
    def get_test_model_cls(self):
        class TestModel(Model):
            def kernel(self, inputs):
                self.inputs_spy = inputs
                self.scope_spy = tf.get_variable_scope()

        return TestModel

    def assertDictIs(self, first, second):
        self.assertEqual(len(first), len(second), 'Dict has differnt len')
        for k in first:
            self.assertIs(first[k], second[k])

    def test_normal_inputs(self):
        x = Constant(1.0, 'x')
        m = self.get_test_model_cls()('test', {'input': x})
        self.assertDictIs(m.inputs_spy, {'input': x})

    def test_single_input(self):
        x = Constant(1.0, 'x')
        m = self.get_test_model_cls()('test', x)
        self.assertDictIs(m.inputs_spy, {'input': x})
