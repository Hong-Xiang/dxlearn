from dxl.data.function import Function
from functools import singledispatch
from dxl.learn.core.global_ctx import get_global_context
import tensorflow as tf
from dxl.learn.core import Tensor


class DropOut(Function):
    def __init__(self, keep_prob=None):
        """
        `keep_prob`: keep probability during training.
        """
        if keep_prob is not None:
            raise NotImplementedError(
                "Dropout with custom keep prob is not implemented yet.")
        self.keep_prob = keep_prob

    def __call__(self, x):
        gtx = get_global_context()
        keep_prob = gtx.tensors[gtx.KEYS.TENSOR.KEEP_PROB]
        return Tensor(tf.nn.dropout(x.data, keep_prob.data))
