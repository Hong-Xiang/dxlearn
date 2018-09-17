from doufo.tensor import Tensor
import tensorflow as tf
import numpy as np
from doufo import singledispatch
from dxl.learn.function.crop import shape_as_list
import copy


@singledispatch(nargs=3, nouts=1)
def data_splitter(x, nb_split, name=None):
    raise NotImplementedError('{} is not supported.'.format(type(x)))


@data_splitter.register(tf.Tensor)
def _(x, nb_split, name='data_splitter'):
    shape_x = shape_as_list(x)
    split_size = shape_x[0] // nb_split
    res = {}
    offset = [0 for i in shape_x]
    slice_size = copy.copy(shape_x)
    slice_size[0] = split_size
    for i in range(nb_split - 1):
        res['slice{}'.format(i)] = tf.slice(x, offset, slice_size)
        offset[0] = offset[0] + split_size
    slice_size[0] = shape_x[0] - offset[0]
    res['slice{}'.format(nb_split - 1)] = tf.slice(x, offset, slice_size)
    return res


@data_splitter.register(Tensor)
def _(x, nb_split, name='data_splitter'):
    x = x.unbox()
    shape_x = shape_as_list(x)
    split_size = shape_x[0] // nb_split
    res = {}
    offset = [0 for i in shape_x]
    slice_size = copy.copy(shape_x)
    slice_size[0] = split_size
    for i in range(nb_split - 1):
        res['slice{}'.format(i)] = Tensor(tf.slice(x, offset, slice_size))
        offset[0] = offset[0] + split_size
    slice_size[0] = shape_x[0] - offset[0]
    res['slice{}'.format(nb_split - 1)] = Tensor(tf.slice(x, offset, slice_size))
    return res


@data_splitter.register(np.ndarray)
def _(x, nb_split, name='data_splitter'):
    shape_x = shape_as_list(x)
    split_size = shape_x[0] // nb_split
    res = {}
    offset = 0
    for i in range(nb_split - 1):
        res['slice{}'.format(i)] = x[offset:offset + split_size]
        offset = offset + split_size
    res['slice{}'.format(nb_split - 1)] = x[offset:]
    return res
