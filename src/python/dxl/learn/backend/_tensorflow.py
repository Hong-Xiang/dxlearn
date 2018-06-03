from .common import Backend
from functools import wraps
import tensorflow as tf
from contextlib import contextmanager


class TensorFlow(Backend):
    def in_sandbox(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            with self.sandbox():
                return func(*args, **kwargs)

        return inner

    @contextmanager
    def sandbox(self):
        with tf.Graph().as_default():
            yield

    def unbox(self):
        return tf

    def _unbox_kernel(self, t):
        from dxl.learn.core import Tensor
        if t is None:
            return None
        if isinstance(t, Tensor):
            return t.unbox()
        return t
        # raise TypeError("Can not unbox {}.".format(t))

    def maybe_unbox(self, t):
        from dxl.learn.utils.general import generic_map
        if isinstance(t, (list, tuple, dict, set)):
            return generic_map(self._unbox_kernel, t)
        else:
            return self._unbox_kernel(t)

    @classmethod
    def TestCase(cls):
        return tf.test.TestCase