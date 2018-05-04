# _*_ coding: utf-8 _*_
import tensorflow as tf
import numpy as np
from typing import List
from ..core import Tensor


def shape_as_list(input_) -> List[int]:
    '''shape as list
    Args:
        input_: Tensor/tf.Tensor/tf.Variable/np.ndarray
    Return: shape of input_ as list.
        if input_ is a list, then return it directly
    '''
    if isinstance(input_, (Tensor, tf.Tensor, tf.Variable)):
        return list(input_.shape.as_list())
    if isinstance(input_, np.ndarray):
        return list(input_.shape)
    if isinstance(input_, (tuple, list)):
        return list(input_)


def random_crop_offset(input_shape, target_shape):
    '''random crop offset
    Args:
        input_shape: list/tuple.
            (batch, width, height, channel) or (width, height, channel)
        target_shape: list/tuple.
            (batch, width, height, channel) or (width, height, channel)
        batched: True or False
    '''
    max_offset = [s - t for s, t in zip(input_shape, target_shape)]
    if any(map(lambda x: x < 0, max_offset)):
        raise ValueError("Invalid input_shape {} or target_shape {}.".format(
            input_shape, target_shape))

    offset = []
    for s in max_offset:
        if s == 0:
            offset.append(0)
        else:
            offset.append(np.random.randint(0, s))
    return np.array(offset)
    

def random_crop(input_, target, name='random_crop'):
    '''random crop
    Args:
        input_: input tensor/Tensor/numpy.
        target: list/tuple/Tensor/numpy/tf.Tensor contains: 
            (batch, width, height, channel) or (width, height, channel)
        name: a name for this operation
    Ｒeturns:
        A cropped tensor of the same rank as input_ and shape target_shape
    '''
    with tf.name_scope(name):
        input_shape = shape_as_list(input_)
        target_shape = shape_as_list(target)
        random_offset = tf.py_func(random_crop_offset,
                                   [input_shape, target_shape], tf.int64)
        
        return tf.slice(input_, random_offset, target_shape)


def align_crop(input_, target, offset=None, name='align_crop'):
    '''align crop
    Args:
        input_: A input tensor/Tensor/numpy.
        target: A list/tuple/Tensor/numpy/tf.Tensor contains: 
            (batch, width, height, channel) or (width, height, channel)
        offset: A list.
        name: A name for this operation
    Ｒeturns:
        A cropped tensor of the same rank as input_ and shape target_shape
    '''
    with tf.name_scope(name):
        shape_input =  shape_as_list(input_)
        shape_output = shape_as_list(target)
        if offset is None:
            offset = [0] + [(shape_input[i] - shape_output[i]) // 2 
                            for i in range(1, 3)] + [0]
        shape_output[3] = shape_input[3]

        return tf.slice(input_, offset, shape_output)


def boundary_crop(input_, offset=None, name="boundary_crop"):
    '''boundary crop
    Args:
        input_: A input tensor/Tensor/numpy.
        offset: A list.
            (batch_offset, width_offset, height_offset, channel_offset)
        name: A name for this operation
    Ｒeturns:
        A cropped tensor of the same rank as input_ and shape target_shape
    '''
    with tf.name_scope(name):
        shape = shape_as_list(input_)
        if len(offset) == 2:
            offset = [0] + list(offset) + [0]
        shape_output = [s - 2 * o for s, o in zip(shape, offset)]
        return tf.slice(input_, offset, shape_output)


