import tensorflow as tf
import numpy as np
from dxl.learn.core import tensor
from dxl.learn.core.graph_info import GraphInfo


class VariableTest(tf.test.TestCase):
  def test_init_constant(self):
    with self.test_session() as sess:
      vinfo = tensor.VariableInfo(initializer=1.0)
      ginfo = GraphInfo('x')
      x = tensor.Variable(vinfo, ginfo)
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(x.eval(), 1.0)


class FuncVariableTest(tf.test.TestCase):
  def test_basic(self):
    x_ = np.array([1.0, 2.0, 3.0], np.float32)
    x = tensor.variable(GraphInfo('x'), initializer=x_)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(x.eval(), [1.0, 2.0, 3.0])