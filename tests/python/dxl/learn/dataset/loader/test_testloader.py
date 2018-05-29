import unittest

from dxl.learn.dataset import TestLoader
from dxl.learn.utils.test_utils import sandbox
import tensorflow as tf


class TestTestLoader(tf.test.TestCase):
    def make_dataset_with_capacity(self):
        capacity = 10
        return TestLoader('dataset/test', capacity=capacity), capacity

    def test_capacity(self):
        dataset, capacity = self.make_dataset_with_capacity()
        self.assertEqual(dataset.capacity, capacity)

    def test_index(self):
        dataset, capacity = self.make_dataset_with_capacity()
        for i in range(dataset.capacity):
            assert dataset[i] == i
