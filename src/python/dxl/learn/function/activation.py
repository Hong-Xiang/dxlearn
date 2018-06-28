from dxl.data.function import function
import tensorflow as tf

__all__ = ['ReLU']


@function
def ReLU(x):
    if isinstance(x, tf.Tensor):
        return tf.nn.relu(x)
    raise NotImplementedError(
        "ReLU(x) not implemented for {}.".format(type(x)))
