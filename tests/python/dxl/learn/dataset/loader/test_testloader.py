import unittest

# from dxl.learn.dataset import TestLoader
from dxl.learn.test import TestCase
import tensorflow as tf
import pytest

@pytest.mark.skip(reason='not impl yet')
class TestTestLoader(TestCase):
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
