import unittest
from dxl.learn.backend import TensorFlow
import tensorflow as tf
import pytest


class TestTensorFlowBackend(tf.test.TestCase):
    def create_dummy_variable(self):
        with tf.variable_scope('scope', reuse=False):
            x = tf.get_variable('x', [])

    def test_create_dummpy_variable_twice(self):
        with tf.Graph().as_default():
            self.create_dummy_variable()
            with pytest.raises(ValueError):
                self.create_dummy_variable()

    def test_sandbox(self):
        backend = TensorFlow()

        @backend.in_sandbox
        def create_variable():
            self.create_dummy_variable()

        create_variable()
        create_variable()