from dxl.learn.dataset.zoo import MNISTColumns
from dxl.learn.test import TestCase
import unittest


class TestMNISTColumns(TestCase):
    def test_train(self):
        c = MNISTColumns(self.resource_path() / 'dataset' / 'mnist.h5', True)
        assert set(c.columns) == {'image', 'label'}
        assert c.capacity == 60000
        c.close()

    def test_test(self):
        c = MNISTColumns(self.resource_path() / 'dataset' / 'mnist.h5', False)
        assert set(c.columns) == {'image', 'label'}
        assert c.capacity == 10000
        c.close()
