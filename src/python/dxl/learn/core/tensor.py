from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

from dxl.fs import Path

from .distribute import Host
from .graph_info import DistributeGraphInfo, GraphInfo
import warnings


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

    def __init__(self,
                 data: tf.Tensor,
                 data_info: DataInfo = None,
                 graph_info: GraphInfo = None):
        self.data_info = data_info
        self.graph_info = graph_info
        self.data: tf.Tensor = self._process_input_data(data)
        if self.graph_info.name is None:
            self.graph_info.set_name(self.data.name)
        self._nb_copied = 0

    def _process_input_data(self, data):
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
        from .session import ThisSession
        return ThisSession.run(self.data)

    def matmul(self, m, constructor=None):
        if constructor is None:
            constructor = lambda d: Tensor(d, self.data_info, self.graph_info.update(name=None))
        d = tf.matmul(self.data, m.data)
        return constructor(d)

    def __matmul__(self, m):
        return self.matmul(m)

    def eval(self):
        return self.data.eval()

    def copy_to(self, host: Host, is_return_variable=False) -> 'Tensor':
        # if host == self.graph_info.host:
        # raise ValueError("Can not copy to original host.")
        self._nb_copied += 1
        name = self.graph_info.name + '_copy_{}'.format(self._nb_copied)
        with self.graph_info.variable_scope(host=host) as scope:
            if self.data_info is None:
                info = None
            else:
                info = self.data_info.info
            vi = VariableInfo(info, self.data.shape, self.data.dtype)
            variable = Variable(vi,
                                self.graph_info.update(
                                    name=name, host=host,
                                    variable_scope=scope))
            assigned = variable.assign(self)
            if is_return_variable:
                return assigned, variable
            return assigned

    @classmethod
    def from_(cls, t: 'Tensor'):
        # with t.graph_info.variable_scope() as scope:
        #     data = tf.identity(t.data, name=t.graph_info.name)
        #     return cls(data=data, data_info=t.data_info, graph_info=t.graph_info.update(name=None))
        return cls(data=t.data, data_info=t.data_info, graph_info=t.graph_info)


class TensorNumpyNDArray(Tensor):
    def _process_input_data(self, data):
        with self.graph_info.variable_scope():
            data = tf.constant(np.array(data), name=self.graph_info.name)
        return data


Constant = TensorNumpyNDArray


class SparseMatrix(Tensor):
    """
    data is required to be scipy.sparse.coo_matrix
    """

    def _process_input_data(self, data):
        with self.graph_info.variable_scope():
            data = tf.SparseTensor(
                np.array([data.row, data.col]).T, data.data, data.shape)
        return data

    def matmul(self, m, constructor=None):
        if constructor is None:
            constructor = lambda d: Tensor(d, self.data_info, self.graph_info.update(name=None))
        d = tf.sparse_tensor_dense_matmul(self.data, m.data)
        return constructor(d)


class VariableInfo(DataInfo):
    def __init__(self, info=None, shape=None, dtype=None, initializer=None):
        super().__init__(info)
        self.shape = shape
        self.dtype = dtype
        self.initializer = initializer


class Variable(Tensor):
    def __init__(self, data_info: VariableInfo, graph_info: GraphInfo):
        super().__init__(None, data_info, graph_info)

    def _is_constant_initializer(self):
        with_init = self.data_info.initializer is not None
        if with_init and isinstance(self.data_info.initializer,
                                    (float, int, np.ndarray)):
            return True
        return False

    def _process_input_data(self, data):
        with self.graph_info.variable_scope():
            name = self.graph_info.name
            if self._is_constant_initializer():
                return tf.get_variable(
                    name, initializer=self.data_info.initializer)
            return tf.get_variable(
                name,
                dtype=self.data_info.dtype,
                shape=self.data_info.shape,
                initializer=tf.initializers.zeros)

    def assign(self, t: Tensor):
        with self.graph_info.variable_scope() as scope:
            if isinstance(t, (np.ndarray, tf.Tensor)):
                data = self.data.assign(t)
            else:
                data = self.data.assign(t.data)
            return Tensor(
                data,
                DataInfo(self.data_info.info),
                self.graph_info.update(name=None))


class TensorVariable:
    def __init__(self, data_info, graph_info):
        warnings.warn(
            "TensorVariable is going to be deprecated, use Variable instead.")
        super().__init__(data_info, graph_info)


def variable(graph_info,
             variable_info=None,
             shape=None,
             dtype=None,
             initializer=None):
    return Variable(
        VariableInfo(variable_info, shape, dtype, initializer), graph_info)


class TensorRaw(Tensor):
    def __add__(self, t: Tensor):
        if isinstance(t, Tensor):
            data = t.data
        elif isinstance(t, tf.Tensor):
            data = t
        else:
            raise TypeError("Required Tensor or tf.Tensor to add.")
        result = self.data + data
        return Tensor(
            result,
            self.data_info,
            self.graph_info.from_dict(
                self.graph_info.update_to_dict(name=result.name)))


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
