from dxl.data.function import Function, shape_list
import numpy as np
import tensorflow as tf
import cntk


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

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            result = np.zeros(shape_list(x) + [self.nb_classes])
            result[np.arange(x.size), x] = 1
            return result
        if isinstance(x, cntk.Variable):
            return cntk.one_hot(x, self.nb_classes)
        if isinstance(x, tf.Tensor):
            return tf.keras.backend.one_hot(x, self.nb_classes)
