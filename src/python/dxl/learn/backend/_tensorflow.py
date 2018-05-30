from .common import Backend
from functools import wraps
import tensorflow as tf


class TensorFlow(Backend):
    def sandbox(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            with tf.Graph().as_default():
                return func(*args, **kwargs)

        return inner

    def unbox(self):
        return tf

    @classmethod
    def TestCase(cls):
        return tf.test.TestCase