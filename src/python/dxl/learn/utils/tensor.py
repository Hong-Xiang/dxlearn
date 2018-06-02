import numpy as np
import tensorflow as tf
import numpy as np


def ensure_shape(input_tensor, shape=None, *, batch_size=None):
    """
    Reshpe input_tensor to given shape.
    """
    if isinstance(input_tensor, np.ndarray):
        return np.copy(input_tensor)
    with tf.name_scope('shape_ensure'):
        shape_input = input_tensor.shape.as_list()
        if shape is None:
            shape = shape_input
        if shape[0] is None:
            shape[0] = batch_size
        return tf.reshape(input_tensor, shape)


import typing


def to_tensor(input_: typing.TypeVar('TensorConvertable', tf.Tensor, np.ndarray),
              *,
              name: str="to_tensor",
              restrict_tensorflow=False,
              default_device='cpu') -> tf.Tensor:
    """
    Helper function to convert multiple kind of Tensor-convertable objects to Tensor.
    """
    if isinstance(input_, Graph):
        return input_.as_tensor()
    if isinstance(input_, np.ndarray) and not restrict_tensorflow:
        return input_
    if isinstance(input_, tf.Tensor):
        return input_
    # with tf.name_scope()


def to_tensor_with_shape(input_, shape: typing.Tuple[int], *, batch_size: int=None, name: str='to_tensor_with_shape') -> tf.Tensor:
    if shape is None and batch_size is not None:
        shape = shape_as_list(input_)
        shape[0] = batch_size

    input_ = to_tensor(input_)
    if isinstance(input_, np.ndarray):
        return input_.reshape(shape)
    elif isinstance(input_, tf.Tensor):
        with tf.name_scope(name):
            return tf.reshape(input_, shape)


def shape_as_list(input_) -> typing.List[int]:
    if isinstance(input_, tf.Tensor):
        return list(input_.shape.as_list())
    if isinstance(input_, np.ndarray):
        return list(input_.shape)
    if isinstance(input_, (tuple, list)):
        return list(input_)

