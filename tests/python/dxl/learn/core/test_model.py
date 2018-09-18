import pytest

from dxl.learn.test import TestCase
# from dxl.learn.model.base import Model, Constant
import tensorflow as tf
import numpy as np

@pytest.mark.skip()
class TestModel(TestCase):
    def get_test_model_cls(self):
        class TestModel(Model):
            def kernel(self, inputs):
                self.inputs_spy = inputs
                self.scope_spy = tf.get_variable_scope()
                return {
                    self.KEYS.TENSOR.OUTPUT: inputs[self.KEYS.TENSOR.INPUT]
                }

        return TestModel

    def get_dummy_input(self):
        return Constant(1.0, 'x')

    def assertDictIs(self, first, second):
        self.assertEqual(len(first), len(second), 'Dict has differnt len')
        for k in first:
            self.assertIs(first[k], second[k])

    def test_normal_inputs(self):
        x = self.get_dummy_input()
        m = self.get_test_model_cls()('test', {'input': x})
        m.make()
        self.assertDictIs(m.inputs_spy, {'input': x})

    def test_single_input(self):
        x = self.get_dummy_input()
        m = self.get_test_model_cls()('test', x)
        m.make()
        self.assertDictIs(m.inputs_spy, {'input': x})

    def test_single_output(self):
        x = self.get_dummy_input()
        m = self.get_test_model_cls()('test', x)
        self.assertIs(m(), x)

    def test_single_output_shortcut(self):
        x = self.get_dummy_input()
        m = self.get_test_model_cls()('test', x)
        y = m(x)
        self.assertNotIsInstance(y, dict)
