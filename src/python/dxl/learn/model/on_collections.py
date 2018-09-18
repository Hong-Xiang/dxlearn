import tensorflow as tf
from ..core import Model, GraphInfo, Tensor, Constant
from dxl.fs import Path

import warnings

warnings.warn(DeprecationWarning("Summation is deprecated, use doufo.tensor.sum_ instead."))

# TODO remove

class Summation(Model):
    def __init__(self, info, inputs=None):
        super().__init__(info, inputs)

    def _maybe_unbox(self, t):
        if isinstance(t, Tensor):
            return t.data
        if isinstance(t, tf.Tensor):
            return t
        raise TypeError("Unable to unbox {} object to tf.Tensor.".format(
            type(t)))

    def kernel(self, inputs=None):
        if inputs is None or len(inputs) == 0:
            return None
        if self.KEYS.TENSOR.INPUT in inputs:
            inputs = inputs[self.KEYS.TENSOR.INPUT]
        if inputs is None or len(inputs) == 0:
            return None
        tf_tensors = []
        if isinstance(inputs, dict):
            for v in inputs.values():
                tf_tensors.append(self._maybe_unbox(v))
        else:
            for v in inputs:
                tf_tensors.append(self._maybe_unbox(v))
        result = tf.add_n(tf_tensors)
        return Tensor(result, self.info.erase_name())
