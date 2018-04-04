import tensorflow as tf
import numpy as np

from unittest.mock import patch, MagicMock

from dxl.learn.graph.reconstruction.master import MasterGraph
from dxl.learn.graph.reconstruction.worker import WorkerGraphBase, WorkerGraphLOR
from dxl.learn.graph.reconstruction.utils import ImageInfo
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


dummyReconStep = MagicMock()


class TestWorkerGraphLOR(tf.test.TestCase):
  def test_basic(self):
    x = np.arange(27, dtype=np.float32).reshape([3, 3, 3])
    m = MasterGraph(x, 2)
    emap = np.ones([3, 3, 3], np.float32)
    lors = {
        'x': np.ones([3, 6], dtype=np.float32),
        'y': np.ones([4, 6], dtype=np.float32),
        'z': np.ones([5, 6], dtype=np.float32)
    }
    image_info = ImageInfo([3, 3, 3], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    w = WorkerGraphLOR(m, image_info,  {
        'x': [3, 6],
        'y': [4, 6],
        'z': [5, 6]
    }, 0)
    w.assign_efficiency_map_and_lors(emap, lors)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(m.tensor(m.KEYS.TENSOR.X).eval(), x)
      self.assertAllEqual(w.tensor(w.KEYS.TENSOR.X).eval(), x)
      sess.run(w.tensor(w.KEYS.TENSOR.INIT).data)
      self.assertAllEqual(w.tensor(w.KEYS.TENSOR.EFFICIENCY_MAP).eval(), emap)


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
