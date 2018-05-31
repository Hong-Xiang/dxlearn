from dxl.learn.core.graph_info import GraphInfo
from dxl.learn.test import TestCase
import tensorflow as tf
import unittest
import re
from functools import wraps


class TestGraphInfo(TestCase):
    def create_simple_info(self, name='x'):
        return GraphInfo(name, None, False)

    def create_reuseable_info(self, name='x'):
        return GraphInfo(name, None, True)

    def test_construct_tensor_info(self):
        info = self.create_simple_info('x')
        self.assertNameEqual(info, 'x')

    def test_scope(self):
        info = self.create_simple_info('scope')
        with info.variable_scope():
            x = tf.get_variable('x', [], tf.float32)
            self.assertNameEqual(x, 'scope/x')

    def test_child_scope(self):
        info = self.create_simple_info('scope')
        cinfo = info.child('sub')
        with info.variable_scope():
            pass
        with cinfo.variable_scope():
            pass
        self.assertNameEqual(info, 'scope')
        self.assertNameEqual(cinfo, 'scope/sub')

    def test_reuse_scope(self):
        info = self.create_simple_info('scope')
        with info.variable_scope():
            x0 = tf.get_variable('x', [], tf.float32)
        with info.variable_scope(reuse=True):
            x1 = tf.get_variable('x', [], tf.float32)
        self.assertNameEqual(x0, 'scope/x')
        self.assertNameEqual(x1, 'scope/x')

    def test_tf_varible_scope_compat(self):
        info = self.create_simple_info('scope')
        with tf.variable_scope('scope0'):
            with info.variable_scope() as scope:
                self.assertNameEqual(scope, 'scope0/scope')

    def test_update(self):
        info = GraphInfo('x', None, False)
        info_u = info.update(name=info.name / 'y')
        self.assertNameEqual(info_u, 'x/y')

    def test_auto_scope(self):
        info = GraphInfo('x', None, False)
        self.assertEqual(info.scope, 'x')

    def test_child(self):
        info = GraphInfo('x', None, False)
        child = info.child('y')
        self.assertNameEqual(child, 'x/y')


if __name__ == "__main__":
    unittest.main()