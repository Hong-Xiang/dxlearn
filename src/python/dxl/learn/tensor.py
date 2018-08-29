from pathlib import Path

import numpy as np
import tensorflow as tf

from dxl.learn.utils.general import strip_colon_and_index_from_name


def copy_to(source, host):
    with scope(host):
        target = copy(source)
        return assign(source, target), target


def no_op(backend=None):
    return tf.no_op()


def constant(backend, data, name=None):
    return tf.constant(data, name)


class SparseTensor(TensorFromExternalData):
    """
    data is required to be scipy.sparse.coo_matrix or a 2-D array.
    If data is a 2-D array, it should has shape [N, ndim+1], data[:, :-1] are coordinates and data[:, -1] are values.
    """

    def _construct_tensor(self, data, name):
        import scipy.sparse
        if isinstance(data, scipy.sparse.coo.coo_matrix):
            data = tf.SparseTensor(
                np.array([data.row, data.col]).T, data.data, data.shape)
        else:
            data = tf.SparseTensor(data[:, :-1], data[:, -1], data.shape)
        return data, GraphInfo(name, tf.get_variable_scope(), False)

    def matmul(self, m, constructor=None):
        if constructor is None:
            def constructor(d): return Tensor(d, self.info.update(name=None))
        d = tf.sparse_tensor_dense_matmul(self.data, m.data)
        return constructor(d)


SparseMatrix = SparseTensor



def initializer(t):
    return t.initializer


from doufo.tensor import array, copy
from dxl.learn.backend import TensorFlowBackend


@array.register(TensorFlowBackend)
@array.register(tf)
def _(backend, shape, dtype, name):
    return tf.get_variable(name, shape, dtype, trainable=False)


def assign(source, target):
    return source.assign(target)


def assign_add(source, target, use_locking=None):
    return tf.assign_add(target, source, use_locking=None)


def variable(backend, name, shape, dtype):


def variable_from_source(backend, name, data):
    return tf.get_variable(name, None, None, data)


def variable_not_trainable(backend, name, shape, dtype, initializer):
    return tf.get_variable(name, shape, dtype, initializer, trainable=False)
