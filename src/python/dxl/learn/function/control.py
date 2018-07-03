from dxl.data.function import Function
import tensorflow as tf
from dxl.learn.core import Tensor

from contextlib import contextmanager

# TODO: refactor to class?


@contextmanager
def ControlDependencies(depens):
    depens_ = [t.data if isinstance(t, Tensor) else t for t in depens]
    with tf.control_dependencies(depens_):
        yield
