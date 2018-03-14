from .base import Host
from typing import TypeVar
import numpy as np
import tensorflow as tf
from dxl.fs import Path
from abc import ABCMeta, abstractmethod


TensorData = TypeVar('TensorData', np.ndarray, tf.Tensor, Path)


class GraphInfo:
    def __init__(self, name, scope, is_save, ):
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

    def __init__(self, data: TensorData, data_info: DataInfo, host: Host, graph_info: GraphInfo):
        self.data = data
        self.data_info = data_info
        self.host = host
        self.graph_info = graph_info
    
    @property
    def shape(self):
        pass
    

    def run(self):
        from .session import ThisSession
        return ThisSession.run(self.data)

    def run_on(self, session):
        pass

    def copy_to(self, host: Host) -> 'Tensor':
        with tf.device(host.device_prefix()):
            with tf.get_variable_scope(self.scope):
                new_name = "{}_{}_{}".format(name, host.job_name, host.task_index)
                data = tf.get_variable(name, shape=self.shape, )


class BroadcastTensor:
    """
    An Tensor collection of broadcasted tensor.
    """
    pass
