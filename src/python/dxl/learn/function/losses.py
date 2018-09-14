import tensorflow as tf
from dxl.learn.model import Model
from doufo import singledispatch
from doufo.tensor import Tensor


@singledispatch(nargs=2, nouts=1)
def mean_square_error(label, infer):
    raise NotImplementedError('{} is not supported.'.format(type(label)))


@mean_square_error.register(tf.Tensor)
def _(label, infer):
    return tf.losses.mean_squared_error(label, infer)


@mean_square_error.register(Tensor)
def _(label, infer):
    if isinstance(infer, Tensor):
        infer = infer.unbox()
    label = label.unbox()
    return Tensor(tf.losses.mean_squared_error(label, infer))


@singledispatch(nargs=2, nouts=1)
def absolute_error(label, infer):
    raise NotImplementedError('{} is not supported.'.format(type(label)))


@absolute_error.register(tf.Tensor)
def _(label, infer):
    return tf.losses.absolute_difference(label, infer)


@absolute_error.register(Tensor)
def _(label, infer):
    if isinstance(infer, Tensor):
        infer = infer.unbox()
    label = label.unbox()
    return Tensor(tf.losses.absolute_difference(label, infer))


@singledispatch(nargs=3, nouts=1)
def poisson_loss(label, data, *, compute_full_loss=False):
    raise NotImplementedError('{} is not supported.'.format(type(label)))


@poisson_loss.register(tf.Tensor)
def _(label, data, *, compute_full_loss=False):
    label = tf.maximum(label, 0.0)
    data = tf.maximum(data, 0.0)
    return tf.reduce_mean(tf.keras.losses.poisson(label, data))


@poisson_loss.register(Tensor)
def _(label, data, *, compute_full_loss=False):
    if isinstance(data, Tensor):
        data = data.unbox()
    label = label.unbox()
    label = tf.maximum(label, 0.0)
    data = tf.maximum(data, 0.0)
    return Tensor(tf.reduce_mean(tf.keras.losses.poisson(label, data)))


@singledispatch(nargs=3, nouts=1)
def log_poisson_loss(log_label, data, *, compute_full_loss=False):
    """
    log_label: log value of expectation (inference)
    data: Poisson sample
    """
    pass


@log_poisson_loss.register(tf.Tensor)
def _(log_label, data, *, compute_full_loss=False):
    data = tf.maximum(data, 0.0)
    return tf.reduce_mean(tf.nn.log_poisson_loss(log_label, data, compute_full_loss))


@log_poisson_loss.register(Tensor)
def _(log_label, data, *, compute_full_loss=False):
    if isinstance(data, Tensor):
        data = data.unbox()
    log_label = log_label.unbox()
    data = tf.maximum(data, 0.0)
    return Tensor(tf.reduce_mean(tf.nn.log_poisson_loss(log_label, data, compute_full_loss)))


@singledispatch(nargs=3, nouts=1)
def composite_loss(label, infer, losses):
    raise NotImplementedError('{} is not supported.'.format(type(label)))


@composite_loss.register(tf.Tensor)
def _(label, infer, losses):
    with tf.variable_scope("composite_loss"):
        weighted_loss = [k(label, infer) * v for k, v in losses.items()]
        return tf.reduce_sum(weighted_loss)


@composite_loss.register(Tensor)
def _(label, infer, losses):
    if isinstance(infer, Tensor):
        infer = infer.unbox()
    label = label.unbox()
    with tf.variable_scope("composite_loss"):
        weighted_loss = [k(label, infer) * v for k, v in losses.items()]
        return Tensor(tf.reduce_sum(weighted_loss))
