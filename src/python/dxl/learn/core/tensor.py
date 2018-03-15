from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

from dxl.fs import Path

from .distribute import Host
from .graph_info import DistributeGraphInfo, GraphInfo


class DataInfo:
    def __init__(self, info):
        self.info = self._unify_data_info(info)

    @classmethod
    def _unify_data_info(cls, data_info: 'DataInfo'):
        if isinstance(data_info, DataInfo):
            return data_info.info
        return data_info


class VariableInfo(DataInfo):
    def __init__(self, info, shape, dtype):
        super().__init__(info)
        self.shape = shape
        self.dtype = dtype


class Tensor:
    """
    Abstract Tensor which is one-to-one mapped to one tensor in tensorflow compute graph. 
    Providing unified interface to `numpy.ndarray`, `tensorflow.Tensor`, hdf5 file on filesystem, etc.
    """

    def __init__(self, data: tf.Tensor, data_info: DataInfo, graph_info: GraphInfo):
        self.data_info = data_info
        self.graph_info = graph_info
        self.data: tf.Tensor = self.process_data(data)
        if self.graph_info.name is None:
            self.graph_info.set_name(self.data.name)
        self._nb_copied = 0

    def process_data(self, data):
        return data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def run(self, session=None):
        if session is not None:
            return session.run(self.data)
        from .session import ThisSession
        return ThisSession.run(self.data)

    def copy_to(self, host: Host, is_return_variable=False) -> 'Tensor':
        if host == self.graph_info.host:
            raise ValueError("Can not copy to original host.")
        self._nb_copied += 1
        name = self.graph_info.name + '_copy_{}'.format(self._nb_copied)
        with self.graph_info.variable_scope(host=host) as scope:
            if self.data_info is None:
                info = None
            else:
                info = self.data_info.info
            vi = VariableInfo(info,
                              self.data.shape, self.data.dtype)
            variable = TensorVariable(vi,
                                      self.graph_info.update(name=name, host=host, variable_scope=scope))
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
    def process_data(self, data):
        with self.graph_info.variable_scope():
            data = tf.constant(np.array(data),
                               name=self.graph_info.name)
        return data


class TensorVariable(Tensor):
    def __init__(self, data_info: VariableInfo, graph_info: GraphInfo):
        super().__init__(None, data_info, graph_info)

    def process_data(self, data):
        with self.graph_info.variable_scope():
            data = tf.get_variable(self.graph_info.name,
                                   dtype=self.data_info.dtype,
                                   shape=self.data_info.shape,
                                   initializer=tf.initializers.zeros)
        return data

    def assign(self, t: Tensor):
        with self.graph_info.variable_scope() as scope:
            data = self.data.assign(t.data)
            return Tensor(data, DataInfo(self.data_info.info), self.graph_info.update(name=None))


class TensorRaw(Tensor):
    def __add__(self, t: Tensor):
        if isinstance(t, Tensor):
            data = t.data
        elif isinstance(t, tf.Tensor):
            data = t
        else:
            raise TypeError("Required Tensor or tf.Tensor to add.")
        result = self.data + data
        return Tensor(result, self.data_info, self.graph_info.from_dict(self.graph_info.update_to_dict(name=result.name)))
