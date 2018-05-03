# import tensorflow as tf
# import numpy as np

# from dxl.learn.graph.reconstruction.master import MasterGraph
# from dxl.learn.core import GraphInfo, Constant


# class TestMasterGraph(tf.test.TestCase):
#   def test_init(self):
#     x = np.array([1.0, 2.0, 3.0], np.float32)
#     g = MasterGraph(x, 2)
#     with self.test_session() as sess:
#       sess.run(tf.global_variables_initializer())
#       self.assertAllEqual(g.tensor(g.KEYS.TENSOR.X).eval(), [1.0, 2.0, 3.0])

#   def test_summation(self):
#     x = np.array([1.0, 2.0, 3.0], np.float32)
#     y = Constant(
#         np.array([3.0, 2.0, 4.0], np.float32), None, GraphInfo('new_value'))
#     g = MasterGraph(x, 2)
#     with self.test_session() as sess:
#       sess.run(tf.global_variables_initializer())
#       buffer = g.tensor(g.KEYS.TENSOR.BUFFER)
#       assigns = [b.assign(y) for b in buffer]
#       sess.run([a.data for a in assigns])
#       sess.run(g.tensor(g.KEYS.TENSOR.UPDATE).data)
#       self.assertAllEqual(g.tensor(g.KEYS.TENSOR.X).eval(), [6.0, 4.0, 8.0])
