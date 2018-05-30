import tensorflow as tf
from dxl.learn.test.fixtures import sandbox


def test_scope_0(sandbox):
    with tf.variable_scope('scope', reuse=False):
        x = tf.get_variable('x', [], tf.float32)


def test_scope_1(sandbox):
    with tf.variable_scope('scope', reuse=False):
        x = tf.get_variable('x', [], tf.float32)