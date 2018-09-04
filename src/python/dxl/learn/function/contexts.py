import contextlib
from doufo import tagfunc
from doufo.tensor import backends
import tensorflow as tf

import tensorflow as tf

__all__ = ['dependencies', 'scope']


@contextlib.contextmanager
def dependencies(xs):
    if all(map(lambda t: isinstance(t, tf.Tensor), xs)):
        with tf.control_dependencies(xs):
            yield
    else:
        raise TypeError(f"Invalid type of xs: {type(xs)}")


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
