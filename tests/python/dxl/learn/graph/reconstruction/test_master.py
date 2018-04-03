import tensorflow as tf
import numpy as np

from dxl.learn.graph.reconstruction.master import MasterGraph
from dxl.learn.core import GraphInfo


class TestMasterGraph(tf.test.TestCase):
  def test_init(self):
    x = np.array([1.0, 2.0, 3.0], np.float32)
    g = MasterGraph(x, 2)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(g.tensor(g.KEYS.TENSOR.X).eval(), [1.0, 2.0, 3.0])
