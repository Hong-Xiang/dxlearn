import contextlib
from doufo import tagfunc
from doufo.tensor import backends
import tensorflow as tf

import tensorflow as tf

__all__ = ['control_dependencies', 'scope']


@contextlib.contextmanager
def control_dependencies(xs):
    if all(map(lambda t: isinstance(t, tf.Tensor))):
        with tf.control_dependencies(xs):
            yield


@tagfunc()
def scope(name, hint=None):
    if hint is not None:
        if isinstance(hint, tf.Tensor):
            return scope[backends.TensorFlowBackend](name)
    raise NotImplementedError


@scope.register(backends.TensorFlowBackend)
@contextlib.contextmanager
def _(name, *args, **kwargs):
    with tf.variable_scope(name, *args, **kwargs):
        yield
