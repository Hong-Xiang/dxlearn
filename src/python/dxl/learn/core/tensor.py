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
        self.data = self.process_data(data)

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

    def copy_to(self, host: Host) -> 'Tensor':
        if host == self.graph_info.host:
            raise ValueError("Can not copy to original host.")
        with self.graph_info.variable_scope(reuse=True, host=host) as scope:
            data = tf.get_variable(name=self.graph_info.name,
                                   shape=self.data.shape,
                                   dtype=self.data.dtype,
                                   collections=[tf.GraphKeys.LOCAL_VARIABLES])
            return Tensor(tf.assign(data, self.data),
                          self.data_info,
                          DistributeGraphInfo(self.graph_info.name, scope,
                                              self.graph_info.reuse, host))

    @classmethod
    def from_(cls, t: 'Tensor'):
        with t.graph_info.variable_scope() as scope:
            data = tf.identity(t.data, name=t.graph_info.name)
            return cls(data=data, data_info=t.data_info, graph_info=t.graph_info)


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
