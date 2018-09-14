from dxl.learn.function.splitter import data_splitter
import tensorflow as tf
from dxl.learn.function.crop import shape_as_list
import numpy as np
from doufo.tensor import Tensor


def test_data_splitter_with_tf_tensor():
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    res1 = data_splitter(x, 4)
    assert all(map(lambda x: shape_as_list(x) == [8, 64, 64, 3], [res1['slice{}'.format(i)] for i in range(4)])) is True


def test_data_splitter_with_tf_tensor_with_larger_last_slice():
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    res2 = data_splitter(x, 5)
    assert all(map(lambda x: shape_as_list(x) == [6, 64, 64, 3], [res2['slice{}'.format(i)] for i in range(4)])) is True
    assert shape_as_list(res2['slice4']) == [8, 64, 64, 3]


def test_data_splitter_with_tensor():
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    x = Tensor(x)
    res1 = data_splitter(x, 4)
    assert all(map(lambda x: shape_as_list(x) == [8, 64, 64, 3], [res1['slice{}'.format(i)] for i in range(4)])) is True
    assert all(map(lambda x: isinstance(x, Tensor), [res1['slice{}'.format(i)] for i in range(4)])) is True


def test_data_splitter_with_tensor_with_larger_last_slice():
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    x = Tensor(x)
    res2 = data_splitter(x, 5)
    assert all(map(lambda x: shape_as_list(x) == [6, 64, 64, 3], [res2['slice{}'.format(i)] for i in range(4)])) is True
    assert all(map(lambda x: isinstance(x, Tensor), [res2['slice{}'.format(i)] for i in range(5)])) is True
    assert shape_as_list(res2['slice4']) == [8, 64, 64, 3]


def test_data_splitter_with_np_array():
    x = np.ones([32, 32, 32, 3], dtype=np.float32)
    res1 = data_splitter(x, 4)
    assert all(map(lambda x: shape_as_list(x) == [8, 32, 32, 3], [res1['slice{}'.format(i)] for i in range(4)])) is True


def test_data_splitter_with_np_array_with_larger_last_slice():
    x = np.ones([32, 32, 32, 3], dtype=np.float32)
    res2 = data_splitter(x, 5)
    assert all(map(lambda x: shape_as_list(x) == [6, 32, 32, 3], [res2['slice{}'.format(i)] for i in range(4)])) is True
    assert shape_as_list(res2['slice4']) == [8, 32, 32, 3]
