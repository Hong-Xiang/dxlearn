from dxl.learn import GraphInfo
import tensorflow as tf
import unittest

def create_simple_info(name='x'):
    return GraphInfo(name, None, False)

def create_reuseable_info(name='x'):
    return GraphInfo(name, None, True)

class TestGraphInfo(unittest.TestCase):
    def test_construct_tensor_info(self):
        info = create_simple_info('x')
        assert info.name = 'x'
    
    def test_scope(self):
        info = create_simple_info('scope')
        with info.variable_scope():
            x = tf.get_variable('x', [], tf.float32)
            assert x.name == 'scope/x'
    
    def test_reuse_scope(self):
        info = create_simple_info('scope')
        with info.variable_scope():
            x0 = tf.get_variable('x', [], tf.float32)
        with info.variable_scope(reuse=True):
            x1 = tf.get_variable('x', [], tf.float32)
        assert x0.name == 'scope/x'
        assert x1.name == 'scope/x'
    
    def test_tf_varible_scope_compat(self):
        info = create_simple_info('scope')
        with tf.variable_scope('scope0'):
            with info.variable_scope() as scope:
                assert scope.name == 'scope0/scope'



        
 