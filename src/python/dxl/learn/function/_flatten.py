from .multi_method_imports import *
from dxl.data.function import shape_list
from dxl.learn.core import Tensor

__all__ = ['Flatten', 'flatten']


class Flatten(Function):
    """
    numpy: tensor.flatten(order='C')

    Keras:
    ```
    keras.layers.Flatten(data_format=None) :: Optional(str) -> Function
    ```

    `data_format` in Keras implemenation is for keep order when switching dataformat (like NHWC to NCHW) 

    ## TensorFlow
    tf.layers.Flatten(data_format, *args, **kwargs)    
    tf.layers.flatten(inputs)

    ## CNTK
    ```
    cntk.ops.flatten(x, axis=None, name="")
    ```



    """

    def __init__(self, batch_dim=0):
        self.batch_dim = batch_dim

    def __call__(self, x):
        if isinstance(x, numpy.ndarray):
            if self.batch_dim == 0:
                return x.reshape([x.shape[0], -1])
            raise NotImplementedError(
                "Flatten for numpy Tensor with batch_dim != 0 is not implemented yet")
            # return numpy.concatenate([x.take(i, self.batch_dim).flatten()]
            #  for i in x.shape[self.batch_dim], axis=self.batch_dim)
        if isinstance(x, tensorflow.Tensor):
            return tensorflow.keras.layers.Flatten()(x)
        if isinstance(x, cntk.Variable):
            return cntk.squeeze(cntk.flatten(x, axis=self.batch_dim))
        if isinstance(x, Tensor):
            return Tensor(self.__call__(x.data))
        raise TypeError("Not supported type of x {}.".format(type(x)))


flatten = Flatten(0)
