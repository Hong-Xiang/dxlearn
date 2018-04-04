import tensorflow as tf
import numpy as np

from dxl.learn.graph.reconstruction.master import MasterGraph
from dxl.learn.graph.reconstruction.worker import WorkerGraphBase
from dxl.learn.core import Constant, Host, Master


class TestWorkerGraphBase(tf.test.TestCase):
  def test_copy_global(self):
    x = np.array([1.0, 2.0, 3.0], np.float32)
    m = MasterGraph(x, 2)
    w = WorkerGraphBase(m, 0)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(m.tensor(m.KEYS.TENSOR.X).eval(), [1.0, 2.0, 3.0])
      self.assertAllEqual(w.tensor(w.KEYS.TENSOR.X).eval(), [1.0, 2.0, 3.0])


# class TestWorkerGraphLOR(tf.test.TestCase):
#   def test_basic(self):
#     x = np.array([1.0, 2.0, 3.0], np.float32)
#     m = MasterGraph(x, 2)
#     emap = np.array([2.0, 4.0, 8.0], np.float32)
#     lors = {
#         'x': np.array([1.0, 3.0], np.float32),
#         'y': np.array([2.0, 4.0], np.float32),
#         'z': np.array([3.0, 6.0], np.float32)
#     }
#     w = WorkerGraphBase(m, None, [3], {'x':[2], 'y':[2], 'z':[2]}, 0)
#     with self.test_session() as sess:
#       sess.run(tf.global_variables_initializer())
#       self.assertAllEqual(m.tensor(m.KEYS.TENSOR.X).eval(), [1.0, 2.0, 3.0])
#       self.assertAllEqual(w.tensor(w.KEYS.TENSOR.X).eval(), [1.0, 2.0, 3.0])


# self.assertAllEqual(g.tensor(g.KEYS.TENSOR.X).eval(), [1.0, 2.0, 3.0])

# import sys
# config = {"master": ["localhost:2333"], "worker": ["localhost:2334"]}
# task = DistributeTask(config)
# task.cluster_init(sys.argv[1], 0)

# def test_cast_0():
#   x = np.array([1.0, 2.0, 3.0], np.float32)
#   m = MasterGraph(x, 2, graph_info=task.ginfo_master())
#   w = WorkerGraphBase(m, h, graph_info=task.ginfo_worker(0))
#   make_distribute_session()
#   print("Expected: [1.0, 2.0, 3.0]")
#   print(m.tensor(m.KEYS.TENSOR.X).run())
# self.assertAllEqual(w.tensor(w.KEYS.TENSOR.X).eval(), [0.0, 0.0, 0.0])
# self.assertAllEqual(
#     w.tensor(w.KEYS.TENSOR.COPY_FROM_GLOBAL).eval(), [1.0, 2.0, 3.0])
# self.assertAllEqual(w.tensor(w.KEYS.TENSOR.X).eval(), [1.0, 2.0, 3.0])
