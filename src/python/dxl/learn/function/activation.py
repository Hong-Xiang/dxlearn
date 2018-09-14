from doufo import singledispatch
import tensorflow as tf
from doufo.tensor import Tensor

__all__ = ['relu', 'selu', 'swish', 'elu', 'celu']


@singledispatch(nargs=1, nouts=1)
def relu(x):
    return x.fmap(relu)


@relu.register(tf.Tensor)
def _(x):
    with tf.variable_scope('relu'):
        return tf.nn.relu(x)


@relu.register(Tensor)
def _(x):
    return x.fmap(relu)


@singledispatch(nargs=1, nouts=1)
def selu(x):
    raise NotImplementedError("SELU not implemented for {}.".format(type(x)))


@selu.register(tf.Tensor)
def _(x):
    with tf.variable_scope('selu'):
        alpha = 1.6732632437728481704
        scale = 1.0507009873554804933
        return scale * tf.where(x >= 0, x, alpha * tf.nn.elu(x))


@selu.register(Tensor)
def _(x):
    return x.fmap(selu)


@singledispatch(nargs=1, nouts=1)
def swish(x):
    return x.fmap(swish)


@swish.register(tf.Tensor)
def _(x):
    with tf.variable_scope('swish', x):
        return x * tf.nn.sigmoid(x)


@swish.register(Tensor)
def _(x):
    return x.fmap(swish)


@singledispatch(nargs=1, nouts=1)
def elu(x):
    raise NotImplementedError("ELU not implemented for {}.".format(type(x)))


@elu.register(tf.Tensor)
def _(x):
    with tf.variable_scope('elu'):
        return tf.nn.elu(x)


@elu.register(Tensor)
def _(x):
    return x.fmap(elu)


@singledispatch(nargs=1, nouts=1)
def celu(x):
    raise NotImplementedError("CELU not implemented for {}".format(type(x)))


@celu.register(tf.Tensor)
def _(x):
    with tf.variable_scope('celu'):
        return tf.nn.elu(tf.concat([x, -x], axis=-1))


@celu.register(Tensor)
def _(x):
    return x.fmap(celu)
