from dxl.data.function import Function
import numpy as np
import tensorflow as tf
import cntk


def backend_of(x):
    if isinstance(x, np.ndarray):
        return np
    if isinstance(x, tf.Tensor):
        return tf
    if isinstance(x, cntk.Variable):
        return cntk
