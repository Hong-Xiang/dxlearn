import tensorflow as tf
import numpy as np
from dxl.learn.core import tensor
from dxl.learn.core.graph_info import GraphInfo
import scipy.sparse


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


class TestSparseMatrix(tf.test.TestCase):
    def test_matmul(self):
        a = np.eye(10)
        b = np.eye(10)
        for i in range(10):
            b[i, i] = i
        a_s = scipy.sparse.coo_matrix(a)
        a_t = tensor.SparseMatrix(a_s, graph_info=GraphInfo('a'))
        b_t = tensor.Constant(b, graph_info=GraphInfo('b'))
        c_t = a_t @ b_t
        with self.test_session() as sess:
            c_t = sess.run(c_t.data)
            np.testing.assert_almost_equal(c_t, a @ b)
