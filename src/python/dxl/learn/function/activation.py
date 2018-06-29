from dxl.data.function import function
import tensorflow as tf
from functools import singledispatch
from dxl.learn.core import Tensor
__all__ = ['ReLU']


@function
def ReLU(x):
    return _ReLU(x)


@singledispatch
def _ReLU(x):
    raise NotImplementedError("Not implemented for {}.".format(type(x)))


@_ReLU.register(Tensor)
def _(x):
    return Tensor(_ReLU(x.data))


@_ReLU.register(tf.Tensor)
def _(x):
    return tf.nn.relu(x)
