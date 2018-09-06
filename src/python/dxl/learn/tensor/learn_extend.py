import numpy as np
import tensorflow as tf

from doufo.tensor import array, copy
from dxl.learn.backend import TensorFlowBackend
from doufo import singledispatch, tagfunc
from dxl.learn.function import scope

__all__ = ['copy_to', 'no_op', 'constant', 'assign', 'assign_add', 'variable', 'variable_from_tensor', 'initializer']


@singledispatch(nargs=2, nouts=2, ndefs=0)
def copy_to(source, host):
    with scope(host):
        target = copy(source)
        return assign(source, target), target


@tagfunc()
def no_op():
    return no_op[TensorFlowBackend]()


@no_op.register(TensorFlowBackend)
def _():
    return tf.no_op()


@tagfunc()
def constant(data, name=None):
    return constant[TensorFlowBackend](data, name)


@constant.register(TensorFlowBackend)
def _(data, name=None):
    return tf.constant(data, name)


# TODO implement sparse

# class SparseTensor(TensorFromExternalData):
#     """
#     data is required to be scipy.sparse.coo_matrix or a 2-D array.
#     If data is a 2-D array, it should has shape [N, ndim+1], data[:, :-1] are coordinates and data[:, -1] are values.
#     """
#
#     def _construct_tensor(self, data, name):
#         import scipy.sparse
#         if isinstance(data, scipy.sparse.coo.coo_matrix):
#             data = tf.SparseTensor(
#                 np.array([data.row, data.col]).T, data.data, data.shape)
#         else:
#             data = tf.SparseTensor(data[:, :-1], data[:, -1], data.shape)
#         return data, GraphInfo(name, tf.get_variable_scope(), False)
#
#     def matmul(self, m, constructor=None):
#         if constructor is None:
#             def constructor(d): return Tensor(d, self.info.update(name=None))
#         d = tf.sparse_tensor_dense_matmul(self.data, m.data)
#         return constructor(d)
#
#
# SparseMatrix = SparseTensor


@singledispatch(nargs=1, nouts=1)
def initializer(t):
    return t.initializer


@array.register(TensorFlowBackend)
@array.register(tf)
def _(shape, dtype, name):
    return tf.get_variable(name, shape, dtype, trainable=False)


@singledispatch(nargs=2)
def assign(source, target):
    return source.assign(target)


@singledispatch(nargs=2, nouts=1, ndefs=1)
def assign_add(source, target, use_locking=None):
    raise NotImplementedError(f"assign_add not implemented for {type(source)}")


@assign_add.register(tf.Tensor)
def _(source, target, use_locking=None):
    return tf.assign_add(target, source, use_locking=None)


@tagfunc(nargs=3, nouts=1, ndefs=1)
def variable(shape, dtype, name=None):
    return np.zeros(shape, dtype)


@variable.register(TensorFlowBackend)
@variable.register(tf)
def _(shape, dtype, name):
    return tf.get_variable(name, shape, dtype)


@tagfunc(nargs=2, nouts=1, ndefs=1)
def variable_from_tensor(data, name=None):
    return np.array(data)


@variable_from_tensor.register(TensorFlowBackend)
@variable_from_tensor.register(tf)
def _(data, name):
    return tf.get_variable(name, initializer=data)
