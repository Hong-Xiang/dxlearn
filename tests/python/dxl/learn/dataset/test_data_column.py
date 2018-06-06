from dxl.learn.dataset import ListColumns, PyTablesColumns
from dxl.learn.test import TestCase
import unittest
from pathlib import Path
from contextlib import contextmanager


class TestListColumns(unittest.TestCase):
    def test_construct(self):
        nb_samples = 100
        x = range(nb_samples)
        y = [(x_**2, x_ + 10) for x_ in x]
        c = ListColumns({'x': x, 'y': y})
        assert c.capacity == 100


class TestPyTablesColumns(unittest.TestCase):
    @contextmanager
    def get_mnist_train_table(self):
        from dxl.learn.test.resource import test_resource_path
        data_path = test_resource_path() / 'dataset' / 'mnist.h5'
        try:
            c = PyTablesColumns(data_path, '/train')
            yield c
        finally:
            c.close()

    def test_mnist_capacity(self):
        with self.get_mnist_train_table() as c:
            assert c.capacity == 60000

    def test_mnist_columns(self):
        with self.get_mnist_train_table() as c:
            assert set(c.columns) == {'image', 'label'}

    def test_mnist_sample(self):
        with self.get_mnist_train_table() as c:
            assert tuple(c[0]['image'].shape) == (28, 28)
            assert tuple(c[0]['label'].shape) == tuple()
