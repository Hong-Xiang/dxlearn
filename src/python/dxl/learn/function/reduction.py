from dxl.data.function import function, Function
from functools import singledispatch
import numpy as np
import tensorflow as tf

__all__ = ['Sum', 'ArgMax']


class Sum(Function):
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, x):
        return _sum(x, self.axis)


@singledispatch
def sum(x, axis=None):
    raise TypeError("Sum not implemented for {}.".format(type(x)))


@sum.register(np.ndarray)
def _(x, axis):
    return np.sum(x, axis)


@sum.register(tf.Tensor)
def _(x, axis):
    return tf.reduce_sum(x, axis=axis)


class ArgMax(Function):
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, x):
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
