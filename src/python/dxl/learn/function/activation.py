from dxl.data.function import function
import tensorflow as tf
from functools import singledispatch
from dxl.learn.core import Tensor
from .scope import Scope
__all__ = ['ReLU', 'SELU', 'Swish', 'ELU']


@function
def ReLU(x):
    return _ReLU(x)


@singledispatch
def _ReLU(x):
    raise NotImplementedError("ReLU not implemented for {}.".format(type(x)))


@_ReLU.register(Tensor)
def _(x):
    return Tensor(_ReLU(x.data))


@_ReLU.register(tf.Tensor)
def _(x):
    return tf.nn.relu(x)

@function
def SELU(x):
    return _SELU(x)

@singledispatch
def _SELU(x):
    raise NotImplementedError("SELU not implemented for {}.".format(type(x)))

@_SELU.register(tf.Tensor)
def _(x):
    with Scope('SELU', x):
        alpha = 1.6732632437728481704
        scale = 1.0507009873554804933
        return scale*tf.where(x>=0, x, alpha*tf.nn.elu(x))

@_SELU.register(Tensor)
def _(x):
    return Tensor(_SELU(x.data))

@function
def Swish(x):
    return _Swish(x)

@singledispatch
def _Swish(x):
    raise NotImplementedError("SELU not implemented for {}.".format(type(x)))

@_Swish.register(tf.Tensor)
def _(x):
    return x * tf.nn.sigmoid(x)

@_Swish.register(Tensor)
def _(x):
    return Tensor(_Swish(x.data))

@function
def ELU(x):
    return _ELU(x)

@singledispatch
def _ELU(x):
    raise NotImplementedError("ELU not implemented for {}.".format(type(x)))

@_ELU.register(tf.Tensor)
def _(x):
    return tf.nn.elu(x)

@_ELU.register(Tensor)
def _(x):
    return Tensor(_ELU(x.data))