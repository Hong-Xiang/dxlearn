from dxl.data.function import Function, shape_list
import numpy as np
import tensorflow as tf
import cntk
from functools import singledispatch
from dxl.learn.core import Tensor


class OneHot(Function):
    """
    numpy:
    Keras: keras.backend.one_hot(indices, num_classes) :: Tensor<n, [#, ...]> -> int -> Tensor<n+1, [#, ...]>
    CNTK: cntk.ops.one_hot(x, num_classes, sparse_output=False, axis=-1, name='') :: Tensor -> int -> bool -> Optional[int] -> Optional[str] -> Tensor | SparseTensor
    TensorFlow: tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
    Pytorch: None
    mxnet: one_hot(indices, depth=_Null, on_value=_Null, off_value=_Null, dtype=_Null, name=None, ...)
    TensorLayer tensorlayer.layres.One_Hot_InputLayer(inputs=None, depth=None, on_value=None, off_value=None, axis=None, dtype=None)
    """

    def __init__(self, nb_classes):
        self.nb_classes = nb_classes

    @singledispatch
    def __call__(self, x):
        return _one_hot(x, self.nb_classes)


@singledispatch
def _one_hot(x, nb_classes):
    raise NotImplementedError("Not implemented for {}.".format(type(x)))


@_one_hot.register(np.ndarray)
def _(x, nb_classes):
    result = np.zeros(shape_list(x) + [nb_classes])
    result[np.arange(x.size), x] = 1
    return result


@_one_hot.register(cntk.Variable)
def _(x, nb_classes):
    return cntk.one_hot(x, nb_classes)


@_one_hot.register(tf.Tensor)
def _(x, nb_classes):
    return tf.keras.backend.one_hot(x, nb_classes)


@_one_hot.register(Tensor)
def _(x, nb_classes):
    return Tensor(_one_hot(x.data, nb_classes))
