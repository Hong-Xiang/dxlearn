# _*_ coding: utf-8 _*_
import tensorflow as tf
import numpy as np
from typing import List
from doufo.tensor import Tensor
from doufo import singledispatch

__all__ = ['shape_as_list', 'random_crop_offset', 'random_crop', 'align_crop', 'boundary_crop']


@singledispatch(nargs=1, nouts=1)
def shape_as_list(input_) -> List[int]:
    '''shape as list
    Args:
        input_: Tensor/tf.Tensor/tf.Variable/np.ndarray
    Return: shape of input_ as list.
        if input_ is a list, then return it directly
    '''
    return list(input_)


@shape_as_list.register(tf.Tensor)
def _(x):
    return list(x.shape.as_list())


@shape_as_list.register(tf.Variable)
def _(x):
    return list(x.shape.as_list())


@shape_as_list.register(Tensor)
def _(x):
    x = x.unbox()
    return list(x.shape.as_list())


@shape_as_list.register(np.ndarray)
def _(x):
    return list(x.shape)


def random_crop_offset(input_shape, target_shape):
    '''random crop offset
    Args:
        input_shape: list/tuple.
            (batch, width, height, channel) or (width, height, channel)
        target_shape: list/tuple.
            (batch, width, height, channel) or (width, height, channel)
        batched: True or False
    '''
    if len(input_shape) != len(target_shape):
        raise ValueError("input_shape and target_shape are mismatched.")
    max_offset = [s - t for s, t in zip(input_shape, target_shape)]
    if any(map(lambda x: x < 0, max_offset)):
        raise ValueError("Invalid input_shape {} or target_shape {}.".format(
            input_shape, target_shape))

    offset = []
    for s in max_offset:
        if s == 0:
            offset.append(0)
        else:
            offset.append(np.random.randint(0, s + 1))
    return np.array(offset)


@singledispatch(nargs=3, nouts=1)
def random_crop(input_, target, name=None):
    '''random crop
    Args:
        input_: input tensor/Tensor/numpy.
        target: list/tuple/Tensor/numpy/tf.Tensor contains:
            (batch, width, height, channel) or (width, height, channel)
        name: a name for this operation
    Returns:
        A cropped tensor of the same rank as input_ and shape target_shape
    '''
    raise NotImplementedError('{} is not implemented or supported.'.format(type(input_)))


@random_crop.register(tf.Tensor)
def _(input_, target, name='random_crop'):
    with tf.name_scope(name):
        input_shape = shape_as_list(input_)
        target_shape = shape_as_list(target)
        random_offset = tf.py_func(random_crop_offset,
                                   [input_shape, target_shape], tf.int32)
        return tf.slice(input_, random_offset, target_shape)


@random_crop.register(Tensor)
def _(input_, target, name='random_crop'):
    input_ = input_.unbox()
    with tf.name_scope(name):
        input_shape = shape_as_list(input_)
        target_shape = shape_as_list(target)
        random_offset = tf.py_func(random_crop_offset,
                                   [input_shape, target_shape], tf.int32)
        return Tensor(tf.slice(input_, random_offset, target_shape))


@random_crop.register(np.ndarray)
def _(input_, target, name='random_crop'):
    input_ = np.expand_dims(input_, axis=2) if len(shape_as_list(input_)) == 2 else input_
    if len(shape_as_list(target)) == 2:
        if not isinstance(target, list):
            target = np.expand_dims(target, axis=2)
        else:
            target.extend([1])
    input_shape = shape_as_list(input_)
    target_shape = shape_as_list(target)
    random_offset = random_crop_offset(input_shape, target_shape)
    res_ = None
    if len(input_shape) == 4:
        res_ = input_[random_offset[0]:random_offset[0] + target_shape[0],
               random_offset[1]:random_offset[1] + target_shape[1],
               random_offset[2]:random_offset[2] + target_shape[2],
               random_offset[3]:random_offset[3] + target_shape[3]]
    elif len(input_shape) == 3:
        res_ = input_[random_offset[0]:random_offset[0] + target_shape[0],
               random_offset[1]:random_offset[1] + target_shape[1],
               random_offset[2]:random_offset[2] + target_shape[2]]
    else:
        raise NotImplementedError("not supported dim > 3")
    return np.array(res_)


@singledispatch(nargs=4, nouts=1)
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
    raise NotImplementedError('{} is not supported.'.format(type(input_)))


@align_crop.register(tf.Tensor)
def _(input_, target, offset=None, name='align_crop'):
    with tf.name_scope(name):
        shape_input = shape_as_list(input_)
        shape_output = shape_as_list(target)
        if offset is None:
            offset = [0] + [(shape_input[i] - shape_output[i]) // 2
                            for i in range(1, 3)] + [0]
        shape_output[-1] = shape_input[-1]
        return tf.slice(input_, offset, shape_output)


@align_crop.register(Tensor)
def _(input_, target, offset=None, name='align_crop'):
    input_ = input_.unbox()
    with tf.name_scope(name):
        shape_input = shape_as_list(input_)
        shape_output = shape_as_list(target)
        if offset is None:
            offset = [0] + [(shape_input[i] - shape_output[i]) // 2
                            for i in range(1, 3)] + [0]
        shape_output[-1] = shape_input[-1]
        return Tensor(tf.slice(input_, offset, shape_output))


@align_crop.register(np.ndarray)
def _(input_, target, offset=None, name='align_crop'):
    input_shape = shape_as_list(input_)
    target_shape = shape_as_list(target)
    if offset is None:
        offset = [0] + [(input_shape[i] - target_shape[i]) // 2
                        for i in range(1, 3)] + [0]
    target_shape[-1] = input_shape[-1]
    diff = [a - b for a, b in zip(input_shape, target_shape)]
    if any(map(lambda x: x < 0, diff)):
        raise ValueError('the shape of {} must be subset of {}.'.format(input_shape, target_shape))
    if any(map(lambda x: x < 0, [a - b for a, b in zip(diff, offset)])):
        raise ValueError('the offset {} is invalid.'.format(offset))
    res_ = None
    if len(input_shape) == 4:
        res_ = input_[offset[0]:offset[0] + target_shape[0],
               offset[1]:offset[1] + target_shape[1],
               offset[2]:offset[2] + target_shape[2],
               offset[3]:offset[3] + target_shape[3]]
    else:
        raise NotImplementedError("only support dim == 4")
    return np.array(res_)


@singledispatch(nargs=3, nouts=1)
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
    raise NotImplementedError('{} is not supported.'.format(type(input_)))


@boundary_crop.register(tf.Tensor)
def _(input_, offset=None, name="boundary_crop"):
    with tf.name_scope(name):
        shape = shape_as_list(input_)
        if len(offset) == 2:
            offset = [0] + list(offset) + [0]
        shape_output = [s - 2 * o for s, o in zip(shape, offset)]
        return tf.slice(input_, offset, shape_output)


@boundary_crop.register(Tensor)
def _(input_, offset=None, name="boundary_crop"):
    input_ = input_.unbox()
    with tf.name_scope(name):
        shape = shape_as_list(input_)
        if len(offset) == 2:
            offset = [0] + list(offset) + [0]
        shape_output = [s - 2 * o for s, o in zip(shape, offset)]
        return Tensor(tf.slice(input_, offset, shape_output))


@boundary_crop.register(np.ndarray)
def _(input_, offset=None, name="boundary_crop"):
    shape = shape_as_list(input_)
    if len(offset) == 2:
        offset = [0] + list(offset) + [0]
    shape_output = [s - 2 * o for s, o in zip(shape, offset)]
    res_ = None
    if len(shape) == 4:
        res_ = input_[offset[0]:offset[0] + shape_output[0],
               offset[1]:offset[1] + shape_output[1],
               offset[2]:offset[2] + shape_output[2],
               offset[3]:offset[3] + shape_output[3]]
    else:
        raise NotImplementedError('only support dim == 4')
    return np.array(res_)
