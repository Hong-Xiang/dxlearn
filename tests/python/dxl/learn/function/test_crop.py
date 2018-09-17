from dxl.learn.function.crop import *
import tensorflow as tf
import numpy as np
from doufo.tensor import Tensor
from doufo.tensor.binary import all_close


def test_shape_as_list_with_tf_tensor():
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    assert shape_as_list(x) == [32, 64, 64, 3]


def test_shape_as_list_with_tf_variable():
    x = tf.Variable([[3], [4], [5], [6]], dtype=tf.float32)
    assert shape_as_list(x) == [4, 1]


def test_shape_as_list_with_np_array():
    y = np.ones([32, 64, 64, 3], dtype=np.float32)
    assert shape_as_list(y) == [32, 64, 64, 3]


def test_shape_as_list_with_Tensor():
    x = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    z = Tensor(x)
    assert shape_as_list(z) == [32, 64, 64, 3]


def test_shape_as_list_with_default():
    x = [1, 2, 3, 4]
    assert shape_as_list(x) == [1, 2, 3, 4]


def test_random_crop_offset():
    a = [5, 5, 5, 5]
    b = [1, 2, 3, 4]
    offset = random_crop_offset(a, b)
    assert all(map(lambda x: x >= 0, offset)) is True
    assert offset[0] <= 4
    assert offset[1] <= 3
    assert offset[2] <= 2
    assert offset[3] <= 1


def test_random_crop_with_tf_tensor():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
    b = [2, 2]
    c = [3, 3]
    res1 = random_crop(a, b)
    res2 = random_crop(a, c)
    assert shape_as_list(res1) == [2, 2]
    assert shape_as_list(res2) == [3, 3]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res2 = sess.run(res2)
    assert all_close(res2, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]) is True


def test_random_crop_with_Tensor():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
    b = [2, 2]
    c = Tensor(a)
    res = random_crop(c, b)
    assert shape_as_list(res) == [2, 2]


def test_random_crop_with_np_array_2_dim():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    b = [2, 2]
    c = np.array([[1, 2], [3, 4]])
    res1 = random_crop(a, b)
    res2 = random_crop(a, c)
    assert shape_as_list(res1) == [2, 2, 1]
    assert shape_as_list(res2) == [2, 2, 1]


def test_random_crop_with_np_array_gt_or_equal_3_dim():
    a = np.array([[[1, 11, 111], [2, 22, 222], [3, 33, 333]],
                  [[4, 44, 444], [5, 55, 555], [6, 66, 666]],
                  [[7, 77, 777], [8, 88, 888], [9, 99, 999]]], dtype=np.float32)
    b = [2, 2, 2]
    res = random_crop(a, b)
    assert shape_as_list(res) == [2, 2, 2]


def test_align_crop_with_tf_tensor_with_offset():
    a = tf.ones([3, 64, 64, 3], dtype=tf.float32)
    b = [2, 32, 32, 3]
    offset = [1, 32, 32, 0]
    res1 = align_crop(a, b, offset)
    assert shape_as_list(res1) == [2, 32, 32, 3]


def test_align_crop_with_tf_tensor_with_default():
    a = tf.constant(3 * [[[[1, 11, 111], [2, 22, 222], [3, 33, 333]],
                          [[4, 44, 444], [5, 55, 555], [6, 66, 666]],
                          [[7, 77, 777], [8, 88, 888], [9, 99, 999]]]], dtype=tf.float32)
    b = [1, 1, 1, 1]
    res1 = align_crop(a, b)
    assert shape_as_list(res1) == [1, 1, 1, 3]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res1 = sess.run(res1)
    assert all_close(res1, [[[[5, 55, 555]]]]) is True


def test_align_crop_with_tensor():
    a = tf.ones([3, 64, 64, 3], dtype=tf.float32)
    b = Tensor(a)
    c = [2, 32, 32, 1]
    offset = [1, 32, 32, 0]
    res = align_crop(b, c, offset)
    assert shape_as_list(res) == [2, 32, 32, 3]


def test_align_crop_with_np_array_with_default():
    a = np.array(3 * [[[[1, 11, 111], [2, 22, 222], [3, 33, 333]],
                       [[4, 44, 444], [5, 55, 555], [6, 66, 666]],
                       [[7, 77, 777], [8, 88, 888], [9, 99, 999]]]])
    b = [1, 1, 1, 1]
    res = align_crop(a, b)
    assert all_close(res, [[[[5, 55, 555]]]]) is True


def test_align_crop_with_np_array_with_offset():
    a = np.array(3 * [[[[1, 11, 111], [2, 22, 222], [3, 33, 333]],
                       [[4, 44, 444], [5, 55, 555], [6, 66, 666]],
                       [[7, 77, 777], [8, 88, 888], [9, 99, 999]]]])
    b = [1, 1, 1, 1]
    offset = [2, 2, 2, 0]
    res = align_crop(a, b, offset)
    assert all_close(res, [[[[9, 99, 999]]]]) is True


def test_boundary_crop_with_tf_tensor():
    a = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    offset = [0, 2, 2, 0]
    res = boundary_crop(a, offset)
    assert shape_as_list(res) == [32, 60, 60, 3]


def test_boundary_crop_with_tensor():
    a = tf.ones([32, 64, 64, 3], dtype=tf.float32)
    a = Tensor(a)
    offset = [0, 2, 2, 0]
    res = boundary_crop(a, offset)
    assert shape_as_list(res) == [32, 60, 60, 3]


def test_boundary_crop_with_np_array():
    a = np.array(3 * [[[[1, 11, 111], [2, 22, 222], [3, 33, 333]],
                       [[4, 44, 444], [5, 55, 555], [6, 66, 666]],
                       [[7, 77, 777], [8, 88, 888], [9, 99, 999]]]])
    offset = [0, 1, 1, 0]
    res = boundary_crop(a, offset)
    assert all_close(res, 3 * [[[[5, 55, 555]]]]) is True
