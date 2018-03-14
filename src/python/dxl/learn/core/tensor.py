from .distribute import Host
from typing import TypeVar
import numpy as np
import tensorflow as tf
from dxl.fs import Path
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager


TensorData = TypeVar('TensorData', np.ndarray, tf.Tensor, Path)


class GraphInfo:
    def __init__(self, name=None, scope=None, is_save=None):
        self.name = name
        self.scope = scope
        self.is_save = is_save


class DataInfo:
    def __init__(self, info):
        self.info = info


class Tensor:
    """
    Abstract Tensor which is one-to-one mapped to one tensor in tensorflow compute graph. 
    Providing unified interface to `numpy.ndarray`, `tensorflow.Tensor`, hdf5 file on filesystem, etc.
    """

    def __init__(self, data: tf.Tensor, data_info: DataInfo, host: Host, graph_info: GraphInfo):
        self.data = data
        self.data_info = self._unify_data_info(data_info)
        self.host = host
        self.graph_info = self._unify_graph_info(graph_info)

    def _unify_data_info(self, data_info: DataInfo):
        if isinstance(data_info, dict):
            return DataInfo(data_info)
        return data_info

    def _unify_graph_info(self, graph_info: GraphInfo):
        if isinstance(graph_info, dict):
            return GraphInfo(**graph_info)
        return graph_info

    @contextmanager
    def device_scope(self, host=None):
        if host is None:
            host = self.host
        with tf.device(host.device_prefix()):
            yield

    @contextmanager
    def variable_scope(self, scope=None):
        if scope is None:
            with tf.variable_scope(self.graph_info.scope, default_name="") as scope:
                yield scope
        else:
            with tf.variable_scope(scope) as scope:
                yield scope

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def run(self):
        from .session import ThisSession
        return ThisSession.run(self.data)

    def run_on(self, session):
        pass

    def copy_to(self, host: Host, graph_info: GraphInfo=None) -> 'Tensor':
        if host == self.host:
            raise ValueError("Can not copy to original host.")
        if graph_info is None or graph_info.name is None:
            name = self.graph_info.name
        else:
            name = graph_info.name
        if graph_info is None:
            scope = None
        else:
            scope = graph_info.scope
        with self.device_scope(host):
            with self.variable_scope(scope) as scope:
                data = tf.get_variable(name, shape=self.data.shape,
                                       dtype=self.data.dtype,
                                       collections=[tf.GraphKeys.LOCAL_VARIABLES])
                data = tf.assign(data, self.data)
                return Tensor(data, self.data_info, host,
                              GraphInfo(name,
                                        scope,
                                        self.graph_info.is_save))

    @classmethod
    def from_(cls, t: 'Tensor', name='convert'):
        with self.device_scope():
            with self.variable_scope():
                with tf.get_variable_scope(name) as scope:
                    data = tf.identity(self.data, name=self.graph_info.name)
                    return cls(data=data, data_info=self.data_info, host=self.host,
                               graph_info=GraphInfo(self.graph_info.name,
                                                    scope,
                                                    self.graph_info.is_save))


class TensorNumpyNDArray(Tensor):
    def __init__(self, data: np.ndarray, data_info: DataInfo, host: Host, graph_info: GraphInfo):
        with self.device_scope(host):
            data = tf.constant(np.array(data),
                               name=self._unify_graph_info(graph_info).name)
            super().__init__(data, data_info, host, graph_info)


class TensorVariable(Tensor):
    def __init__(self, shape, dtype, data_info: DataInfo, host: Host, graph_info: GraphInfo):
        with self.device_scope(host):
            graph_info = self._unify_graph_info(graph_info)
            with self.variable_scope(graph_info.scope or ''):
                data = tf.get_variable(graph_info.name, dtype=dtype, shape=shape,
                                       initializer=tf.initializers.zeros)
                super().__init__(data, data_info, host, graph_info)


class BroadcastTensor:
    """
    An Tensor collection of broadcasted tensor.
    """
    pass
