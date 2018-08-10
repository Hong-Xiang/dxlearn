from functools import singledispatch
from contextlib import contextmanager
import numpy as np
import tensorflow as tf


class Scope:
    """
    Unified Scope.

    Useage:

    ```
    with Scope('scope', np.one([3])) as sess:
        pass

    with Scope('name', tf) as sess:
        x = tf.get_variable('x', shape=[])
    y = tf.get_variable('y', shape=[])
    print(x.name) # name/x:0
    print(y.name) # y:0
    ```
    """

    def __init__(self, name, hint, *args, **kwargs):
        self.name = name
        self.scope = _scope(hint, name, *args, **kwargs)

    def __enter__(self):
        return self.scope.__enter__()

    def __exit__(self, value, type, track_back):
        return self.scope.__exit__(value, type, track_back)


@singledispatch
def _scope(hint, name, *args, **kwargs):
    if hint is tf:
        return _scope_tf(hint, name, *args, **kwargs)
    raise TypeError("Unknown hint type: {}".format(type(hint)))


@_scope.register(tf.Tensor)
def _scope_tf(hint, name, *args, **kwargs):
    return tf.variable_scope(name, *args, **kwargs)


@_scope.register(np.ndarray)
def _(hint, name, *args, **kwargs):
    class NumpyDummyScope:
        def __enter__(self):
            pass

        def __exit__(self, v, t, tb):
            pass
    return NumpyDummyScope()
