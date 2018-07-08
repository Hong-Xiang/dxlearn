from dxl.data.function import function, Function
from functools import singledispatch
import numpy as np
import tensorflow as tf
from dxl.learn.core import Tensor

from .scope import Scope
__all__ = ['Sum', 'ArgMax']


class Sum(Function):
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, x):
        name = 'sum' if self.axis is None else 'sum_{}'.format(self.axis)
        with Scope('sum', x):
            return _sum(x, self.axis)


@singledispatch
def _sum(x, axis=None):
    raise TypeError("Sum not implemented for {}.".format(type(x)))


@_sum.register(np.ndarray)
def _(x, axis):
    return np.sum(x, axis)


@_sum.register(tf.Tensor)
def _(x, axis):
    return tf.reduce_sum(x, axis=axis)

@_sum.register(Tensor)
def _(x, axis):
    return _sum(x.data, axis)

class ArgMax(Function):
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, x):
        with Scope('argmax', x):
            return _argmax(x, self.axis)


@singledispatch
def _argmax(x, axis):
    raise TypeError("ArgMax not implemented for {}.".format(type(x)))


@_argmax.register(np.ndarray)
def _(x, axis):
    return np.argmax(x, axis=axis)


@_argmax.register(tf.Tensor)
def _(x, axis):
    return tf.argmax(x, axis=axis)

@_argmax.register(Tensor)
def _(x, axis):
    return _argmax(x.data, axis)