from .base import Host
from typing import TypeVar
import numpy as np
import tensorflow as tf
from dxl.fs import Path


TensorData = TypeVar('TensorData', np.ndarray, tf.Tensor, Path)


class Tensor:
    """
    Abstract Tensor object with unified interface to `numpy.ndarray`, 
    `tensorflow.Tensor`, and hdf5 file on filesystem.
    """

    def __init__(self, data: TensorData, host: Host, name: str or Path, scope=None):
        self.data = data
        self.host = host
        self.name = name
        self.scope = scope

    def copy_to(self, host: Host) -> 'Tensor':
        with tf.device(host.device_prefix()):
            with tf.get_variable_scope(self.scope) as scope:
                scope = scope
                data = tf.get_variable(name=name, )(self.data, name=self.name)
