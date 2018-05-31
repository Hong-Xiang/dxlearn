from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

from pathlib import Path

from .distribute import Host
from .graph_info import DistributeGraphInfo, GraphInfo
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
        if hasattr(data, 'name'):
            name_hint = data.name
        else:
            name_hint = None
        self.info = self.make_info(info, name_hint)
        self.data = self._process_input_data(data)
        self._nb_copied = 0

    def make_info(self, info, name_hint=None):
        def parse_scope_from_name_hint():
            return str(Path(strip_colon_and_index_from_name(name_hint)).parent)

        if info is None:
            return GraphInfo(name_hint, parse_scope_from_name_hint(), None)
        if isinstance(info, (str, Path)):
            return GraphInfo(Path(info), parse_scope_from_name_hint())
        if not isinstance(info, GraphInfo):
            raise TypeError("Invalid info type {}.".format(type(info)))
        if info.name is not None:
            return info
        if info.scope is not None:
            scope = info.scope
        else:
            scope = parse_scope_from_name_hint()
        reuse = info.reuse
        return GraphInfo(name, scope, reuse)

    def _process_input_data(self, data):
        if isinstance(data, Tensor):
            return data.data
        else:
            return data

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

    def copy_to(self, host: Host, is_return_variable=False) -> 'Tensor':
        # if host == self.graph_info.host:
        # raise ValueError("Can not copy to original host.")
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


class NoOp(Tensor):
    def __init__(self, info=None):
        super().__init__(tf.no_op(), info)


class Constant(Tensor):
    def _process_input_data(self, data):
        with self.info.variable_scope():
            data = tf.constant(data, name=str(self.info.name))
        return data

    @classmethod
    def from_config(cls, ndarray_spec, graph_info):
        from dxl.data.io import load_array
        data = load_array(ndarray_spec)
        return cls(data, None, graph_info)


TensorNumpyNDArray = Constant


class SparseTensor(Tensor):
    """
    data is required to be scipy.sparse.coo_matrix or a 2-D array.
    If data is a 2-D array, it should has shape [N, ndim+1], data[:, :-1] are coordinates and data[:, -1] are values.
    """

    def _process_input_data(self, data):
        import scipy.sparse
        with self.info.variable_scope():
            if isinstance(data, scipy.sparse.coo.coo_matrix):
                data = tf.SparseTensor(
                    np.array([data.row, data.col]).T, data.data, data.shape)
            else:
                data = tf.SparseTensor(data[:, :-1], data[:, -1], data.shape)
        return data

    def matmul(self, m, constructor=None):
        if constructor is None:
            constructor = lambda d: Tensor(d, self.info.update(name=None))
        d = tf.sparse_tensor_dense_matmul(self.data, m.data)
        return constructor(d)


SparseMatrix = SparseTensor


class VariableInfo(DataInfo):
    def __init__(self, info=None, shape=None, dtype=None, initializer=None):
        super().__init__(info)
        self.shape = shape
        self.dtype = dtype
        self.initializer = initializer


class Variable(Tensor):
    def __init__(self, info, shape, dtype, initializer=None):
        super().__init__({
            'shape': shape,
            'dtype': dtype,
            'init': initializer
        }, info)

    def _is_constant_initializer(self):
        with_init = self.data_info.initializer is not None
        if with_init and isinstance(self.data_info.initializer,
                                    (float, int, np.ndarray)):
            return True
        return False

    def _process_input_data(self, data):
        with self.info.variable_scope():
            initializer = data['init']
            if initializer is None:
                initializer = tf.initializers.zeros
                shape = data['shape']
            else:
                shape = None
            return tf.get_variable(
                self.info.relative_name(),
                shape=shape,
                dtype=data['dtype'],
                initializer=initializer)

    def assign(self, t: Tensor, info=None):
        if info is None:
            info = self.graph_info
        if isinstance(info, str):
            info = self.graph_info.update(name=info)
        with info.variable_scope() as scope:
            new_name = info.name if not info is self.graph_info else None
            if isinstance(t, (np.ndarray, tf.Tensor)):
                data = self.data.assign(t)
            else:
                data = self.data.assign(t.data)
            return Tensor(data, None, info)

    def init(self):
        return Tensor(self.data.initializer, None, self.graph_info)


def variable(graph_info,
             variable_info=None,
             shape=None,
             dtype=None,
             initializer=None):
    warnings.warn(DeprecationWarning("Use Variable directly."))
    return Variable(graph_info, shape, dtype, initializer)


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
