import tensorflow as tf
import numpy as np
from dxl.learn.test import TestCase
from dxl.learn.core import tensor
from dxl.learn.core.graph_info import GraphInfo
import scipy.sparse


class TestTensor(TestCase):
    def test_parse_scope_from_name_hint(self):
        assert tensor.Tensor._parse_scope_from_name_hint('x:0') == ''

    def test_parse_scope_from_name_hint_2(self):
        assert tensor.Tensor._parse_scope_from_name_hint(
            'scope/x:0') == 'scope'

    def test_parse_scope_from_name_hint_3(self):
        assert tensor.Tensor._parse_scope_from_name_hint('x') == ''

    def test_parse_scope_from_name_hint_4(self):
        assert tensor.Tensor._parse_scope_from_name_hint('scope/x') == 'scope'

    def test_copy_to_result_with_same_type(self):
        from dxl.learn.distribute import Host, DistributeGraphInfo

        class NewTensorType(tensor.Tensor):
            pass

        t_raw = self.make_dummy_tensor()
        h1 = Host('test', 1)
        t_new = NewTensorType(t_raw.data,
                              DistributeGraphInfo.from_local_info(
                                  t_raw.info, h1))
        h = Host('test', 0)
        t_copy = t_new.copy_to(h, maker=NewTensorType)
        self.assertIsInstance(t_copy, NewTensorType)


class TestVariable(TestCase):
    def test_make_info(self):
        x = tensor.Variable('x', [], tf.float32)
        self.assertNameEqual(x.info, 'x')
        assert x.info.scope.name == ''

    def test_name(self):
        x = tensor.Variable('x', [], tf.float32)
        assert x.data.name == 'x:0'

    def test_in_scope_name(self):
        with tf.variable_scope('scope'):
            x = tensor.Variable('x', [], tf.float32)
        assert x.data.name == 'scope/x:0'

    def test_init_constant(self):
        x = tensor.Variable('x', [], tf.float32, 1.0)
        with self.variables_initialized_test_session() as sess:
            self.assertAllEqual(x.eval(), 1.0)

    def test_basic(self):
        x_ = np.array([1.0, 2.0, 3.0], np.float32)
        x = tensor.Variable(GraphInfo('x'), initializer=x_)
        with self.variables_initialized_test_session() as sess:
            self.assertAllEqual(x.eval(), [1.0, 2.0, 3.0])

    def test_construct_by_graph_info_name(self):
        x = tensor.Variable(GraphInfo('x', 'scope', False), initializer=0)
        assert x.data.name == 'scope/x:0'

    def test_construct_by_graph_info_value(self):
        x = tensor.Variable(GraphInfo('x', 'scope', False), initializer=0)
        with self.variables_initialized_test_session() as sess:
            assert sess.run(x.data) == 0


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
    def test_name(self):
        a = tensor.Constant(1.0, 'x')
        assert a.data.name == 'x:0'

    def test_under_variable_scope(self):
        with tf.variable_scope('scope'):
            a = tensor.Constant(1.0, 'x')
        assert a.data.name == 'scope/x:0'
        assert a.info.scope.name == 'scope'

    def test_run(self):
        value = [1.0, 2.0]
        a = tensor.Constant(value, 'x')
        with self.test_session() as sess:
            result = sess.run(a.data)
        self.assertFloatArrayEqual(value, result)
