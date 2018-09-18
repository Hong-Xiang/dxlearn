# from dxl.learn.train.optimizer import RMSPropOptimizer
# from dxl.learn.core import NotTrainableVariable
# from dxl.learn.test import TestCase
# import numpy as np
# import tensorflow as tf
#
# import pytest
#
# import unittest
#
#
# class TestRMSPropOptimizer(TestCase):
#     def test_learning_rate(self):
#         o = RMSPropOptimizer('optimizer', learning_rate=1e-3)
#         o.make()
#         assert isinstance(o.tensors['learning_rate'], NotTrainableVariable)
#         with self.variables_initialized_test_session() as sess:
#             self.assertAlmostEqual(sess.run(o.tensors['learning_rate']), 1e-3)
#
#     def test_learning_rate_decay(self):
#         o = RMSPropOptimizer('optimizer', learning_rate=1e-3)
#         o.make()
#         with self.variables_initialized_test_session() as sess:
#             sess.run(o.tensors['decay_learning_rate'])
#             self.assertAlmostEqual(sess.run(o.tensors['learning_rate']), 1e-4)
#
#     def test_optimization(self):
#         x = tf.placeholder(tf.float32, [None, 1])
#         y = tf.placeholder(tf.float32, [None, 1])
#         h = x
#         for i in range(10):
#             h = tf.layers.dense(h, 32, tf.nn.relu)
#         y_ = tf.layers.dense(h, 1)
#         loss = tf.losses.mean_squared_error(y, y_)
#         opt = RMSPropOptimizer('optim', learning_rate=1e-3)
#         opt.make()
#         ts = opt.minimize(loss)
#
#         def get_data():
#             x = (np.random.rand(32, 1) * 2.0 - 1) * np.pi
#             y = np.sin(x)
#             return x, y
#
#         with self.test_session() as sess:
#             sess.run(tf.global_variables_initializer())
#             xv, yv = get_data()
#             loss_init = sess.run(loss, {x: xv, y: yv})
#             for i in range(500):
#                 xv, yv = get_data()
#                 sess.run(ts, {x: xv, y: yv})
#             xv, yv = get_data()
#             loss_end = sess.run(loss, {x: xv, y: yv})
#             assert loss_init > 0.3
#             assert loss_end < 0.05
