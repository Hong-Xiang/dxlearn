from dxl.learn.dataset import ListColumns, PyTablesColumns, DataColumns, RangeColumns
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


class TestRangeColumns(unittest.TestCase):
    def test_capacity(self):
        c = RangeColumns(10)
        assert c.capacity == 10

    def test_sample(self):
        nb_samples = 10
        c = RangeColumns(nb_samples)
        samples = [s for s in c]
        assert samples == list(range(nb_samples))


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


class TestDataColumnsIterator(unittest.TestCase):
    def test_next(self):
        nb_samples = 10

        class RangeColumns(DataColumns):
            def _make_iterator(self):
                return iter(range(nb_samples))

        c = RangeColumns(None)
        samples = [s for s in c]
        assert len(samples) == nb_samples
        for i in range(nb_samples):
            assert samples[i] == i
