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

    @classmethod
    def TestCase(cls):
        return tf.test.TestCase