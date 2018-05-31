import tensorflow as tf
import numpy as np
from dxl.learn.test import TestCase
from dxl.learn.core import tensor
from dxl.learn.core.graph_info import GraphInfo
import scipy.sparse


class VariableTest(TestCase):
    def test_init_constant(self):
        x = tensor.Variable('x', [], tf.float32, 1.0)
        with self.variables_initialized_test_session() as sess:
            self.assertAllEqual(x.eval(), 1.0)


class FuncVariableTest(TestCase):
    def test_basic(self):
        x_ = np.array([1.0, 2.0, 3.0], np.float32)
        x = tensor.variable(GraphInfo('x'), initializer=x_)
        with self.variables_initialized_test_session() as sess:
            self.assertAllEqual(x.eval(), [1.0, 2.0, 3.0])


class TestSparseMatrix(TestCase):
    def test_matmul(self):
        a = np.eye(10)
        b = np.eye(10)
        for i in range(10):
            b[i, i] = i
        a_s = scipy.sparse.coo_matrix(a)
        a_t = tensor.SparseMatrix(a_s, info=GraphInfo('a'))
        b_t = tensor.Constant(b, info=GraphInfo('b'))
        c_t = a_t @ b_t
        with self.test_session() as sess:
            c_t = sess.run(c_t.data)
            np.testing.assert_almost_equal(c_t, a @ b)


class TestConstant(TestCase):
    def test_run(self):
        value = [1.0, 2.0]
        a = tensor.Constant(value, 'x')
        with self.test_session() as sess:
            result = sess.run(a.data)
        self.assertFloatArrayEqual(value, result)
