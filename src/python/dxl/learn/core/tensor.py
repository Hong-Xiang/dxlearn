from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

from pathlib import Path

from .graph_info import GraphInfo
import warnings

from dxl.learn.utils.general import strip_colon_and_index_from_name


class DataInfo:
    def __init__(self, info):
        self.info = self._unify_data_info(info)

    @classmethod
    def _unify_data_info(cls, data_info: 'DataInfo'):
        if isinstance(data_info, DataInfo):
            return data_info.info
        return data_info


class Tensor:
    """
    Abstract Tensor which is one-to-one mapped to one tensor in tensorflow compute graph. 
    Providing unified interface to `numpy.ndarray`, `tensorflow.Tensor`, hdf5 file on filesystem, etc.
    """

    def __init__(self, data: tf.Tensor, info: GraphInfo = None):
        self.data = self._maybe_unbox(data)
        self.info = self._make_info(info,
                                    strip_colon_and_index_from_name(
                                        self._get_name(self.data, info)))
        # self.data = self._process_input_data(data)
        self._nb_copied = 0

    def _maybe_unbox(self, data):
        if isinstance(data, Tensor):
            return data.data
        else:
            return data

    @classmethod
    def _get_name(self, tensor, info):
        if isinstance(tensor, Tensor):
            return tensor.info.name
        if isinstance(tensor, tf.SparseTensor):
            if isinstance(info, (str, Path)):
                return str(info)
            else:
                return str(info.name)
        return tensor.name

    @classmethod
    def _parse_scope_from_name_hint(cls, name_hint):
        result = str(Path(strip_colon_and_index_from_name(name_hint)).parent)
        if result == '.':
            result = ''
        return result

    def _deal_with_non_graph_info_info(self, info, backend_name):
        hit = False
        if info is None:
            full_name = backend_name
            hit = True
        if isinstance(info, (str, Path)):
            full_name = str(info)
            hit = True
        if not hit:
            raise TypeError("Invalid info type {}.".format(type(info)))
        return GraphInfo(full_name,
                         self._parse_scope_from_name_hint(full_name), None)

    def _make_info(self, info, name_hint=None):
        if not isinstance(info, GraphInfo):
            result = self._deal_with_non_graph_info_info(info, name_hint)
            if result is not None:
                return result
        if info.name is not None:
            return info
        if info.scope is not None:
            scope = info.scope
        else:
            scope = self.parse_scope_from_name_hint(name_hint)
        reuse = info.reuse
        return GraphInfo(name, scope, reuse)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    def run(self, session=None):
        if session is not None:
            return session.run(self.data)
        from .session import default_session
        return default_session().run(self.data)

    def tensor_with_same_info_except_name(self, d):
        if isinstance(d, Tensor):
            return d
        return Tensor(d, self.info.erase_name())

    def matmul(self, m):
        d = tf.matmul(self.data, m.data)
        return self.tensor_with_same_info_except_name(d)

    def __matmul__(self, m):
        return self.matmul(m)

    def __add__(self, x):
        if isinstance(x, Tensor):
            result = self.data + x.data
        else:
            result = self.data + x
        return self.tensor_with_same_info_except_name(result)

    def __sub__(self, x):
        if isinstance(x, Tensor):
            result = self.data - x.data
        else:
            result = self.data - x
        return self.tensor_with_same_info_except_name(result)

    def __truediv__(self, x):
        if isinstance(x, Tensor):
            result = self.data / x.data
        else:
            result = self.data / x
        return self.tensor_with_same_info_except_name(result)

    def __mod__(self, x):
        if isinstance(x, Tensor):
            result = self.data % x.data
        else:
            result = self.data % x
        return self.tensor_with_same_info_except_name(result)

    def eval(self):
        return self.data.eval()

    def copy_to(self, host: 'Host', is_return_variable=False) -> 'Tensor':
        # if host == self.graph_info.host:
        # raise ValueError("Can not copy to original host.")
        from ..distribute import Host
        self._nb_copied += 1
        name = Path(str(self.info.name) + '_copy_{}'.format(self._nb_copied))
        with self.info.variable_scope(host=host) as scope:
            variable = Variable(
                self.info.update(name=name, host=host, variable_scope=scope),
                self.shape,
                self.dtype,
            )
            assigned = variable.assign(self)
            if is_return_variable:
                return assigned, variable
            return assigned

    def transpose(self, perm=None, name='transpose', conjugate=False):
        result = tf.transpose(self.data, perm, name, conjugate)
        return self.tensor_with_same_info_except_name(result)

    @classmethod
    def from_(cls, t: 'Tensor'):
        warnings.warn(DeprecationWarning('Use construct directly.'))
        # with t.graph_info.variable_scope() as scope:
        #     data = tf.identity(t.data, name=t.graph_info.name)
        #     return cls(data=data, data_info=t.data_info, graph_info=t.graph_info.update(name=None))
        return cls(data=t.data, data_info=t.data_info, graph_info=t.graph_info)


class TensorFromExternalData(Tensor):
    def __init__(self, data, info):
        data, info = self._make_data_and_info(data, info)
        super().__init__(data, info)

    def _make_data_and_info(self, data, info):
        if isinstance(info, (str, Path)):
            return self._construct_tensor(self._parse_data(data), str(info))

        if isinstance(info, GraphInfo):
            with info.variable_scope():
                return self._construct_tensor(
                    self._parse_data(data), str(info.name))
        raise TypeError("Invalid info: {}.".format(info))

    def _parse_data(self, data):
        return data

    def _construct_tensor(self, data, name):
        raise NotImplementedError


class NoOp(Tensor):
    def __init__(self, info=None):
        super().__init__(tf.no_op(), info)


class Constant(TensorFromExternalData):
    def _construct_tensor(self, data, name):
        return tf.constant(
            data, name=name), GraphInfo(name, tf.get_variable_scope(), False)


class SparseTensor(TensorFromExternalData):
    """
    data is required to be scipy.sparse.coo_matrix or a 2-D array.
    If data is a 2-D array, it should has shape [N, ndim+1], data[:, :-1] are coordinates and data[:, -1] are values.
    """

    def _construct_tensor(self, data, name):
        import scipy.sparse
        if isinstance(data, scipy.sparse.coo.coo_matrix):
            data = tf.SparseTensor(
                np.array([data.row, data.col]).T, data.data, data.shape)
        else:
            data = tf.SparseTensor(data[:, :-1], data[:, -1], data.shape)
        return data, GraphInfo(name, tf.get_variable_scope(), False)

    def matmul(self, m, constructor=None):
        if constructor is None:
            constructor = lambda d: Tensor(d, self.info.update(name=None))
        d = tf.sparse_tensor_dense_matmul(self.data, m.data)
        return constructor(d)


SparseMatrix = SparseTensor


class Variable(Tensor):
    def __init__(self, info, shape=None, dtype=None, initializer=None):
        shape, dtype, initializer = self._unified_shape_dtype_and_initializer(
            shape, dtype, initializer)
        data, info = self._make_data_and_info(shape, dtype, initializer, info)
        super().__init__(data, info)

    def _make_data_and_info(self, shape, dtype, initializer, info):
        if isinstance(info, (str, Path)):
            return tf.get_variable(str(info), shape, dtype,
                                   initializer), GraphInfo(
                                       info, tf.get_variable_scope(), False)
        if not isinstance(info, GraphInfo):
            raise TypeError("Invalid info type {}.".format(type(info)))
        with info.variable_scope():
            return tf.get_variable(str(info.name), shape, dtype,
                                   initializer), info

    def _unified_shape_dtype_and_initializer(self, shape, dtype, initializer):
        if initializer is not None and isinstance(initializer,
                                                  (float, int, np.ndarray)):
            shape, dtype = None, None
        else:
            if shape is None:
                shape = initializer.shape
            if dtype is None:
                dtype = initializer.dtype
        return shape, dtype, initializer

    # def _is_constant_initializer(self):
    #     with_init = self.data_info.initializer is not None
    #     if with_init and isinstance(self.data_info.initializer,
    #                                 (float, int, np.ndarray)):
    #         return True
    #     return False

    # def _process_input_data(self, data):
    #     with self.info.variable_scope():
    #         initializer = data['init']
    #         if initializer is None:
    #             initializer = tf.initializers.zeros
    #             shape = data['shape']
    #         else:
    #             shape = None
    #         return tf.get_variable(
    #             self.info.name.name,
    #             shape=shape,
    #             dtype=data['dtype'],
    #             initializer=initializer)

    def assign(self, t: Tensor, info=None):
        if info is None:
            info = self.info
        if isinstance(info, (str, Path)):
            info = self.info.update(name=info)
        with info.variable_scope() as scope:
            new_name = info.name if not info is self.info else None
            if isinstance(t, (np.ndarray, tf.Tensor)):
                data = self.data.assign(t)
            else:
                data = self.data.assign(t.data)
            return Tensor(data, info)

    def init(self):
        return Tensor(self.data.initializer, self.info.erase_name())


def tf_tensor(t: Tensor):
    """
    Unified access to convert tensor to Tensor of tensorflow.
    """
    if isinstance(t, tf.Tensor):
        return t
    if isinstance(t, Tensor):
        return t.data
    if isinstance(t, np.ndarray):
        if t.dtype == np.float64:
            t = t.astype(np.float32)
        return tf.constant(t, name="from_numpy_ndarray")
    raise TypeError("Can not convert {} to {}".format(type(t), tf.Tensor))
