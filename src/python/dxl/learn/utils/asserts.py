import tensorflow as tf
import numpy as np
from dxpy.code.exceptions import DXPYTypeError
from .tensor import shape_as_list


def assert_scalar(tensor):
    if isinstance(tensor, tf.Tensor):
        shape = shape_as_list(tensor)
        if len(shape) > 1 or (len(shape) == 1 and shape[0] > 1):
            msg = "Assert scalar failed for tensor with shape{}."
            raise ValueError(msg.format(shape))
