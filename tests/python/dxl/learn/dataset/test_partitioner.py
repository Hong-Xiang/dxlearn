from dxl.learn.dataset.partitioner import Partitioner, CrossValidatePartitioner, Train80Partitioner
from dxl.learn.dataset.data_column import DataColumns
import unittest
import pytest


class PartitionerTestCase(unittest.TestCase):
    def get_data_column(self, nb_samples):
        class RangeDataColumn(DataColumns):
            def __init__(self, nb_samples):
                super().__init__(range(nb_samples))
                self._nb_samples = nb_samples

            def _calculate_capacity(self):
                return self._nb_samples

            def __getitem__(self, i):
                if i >= self._nb_samples:
                    raise ValueError('{} exceed nb_sampels {}.'.format(
                        i, self._nb_samples))
                return i

        return RangeDataColumn(nb_samples)


class TestPartition(PartitionerTestCase):
    def test_next(self):
        nb_samples = 10
        p = Partitioner().partition(self.get_data_column(nb_samples))
        for i in range(nb_samples):
            assert next(p) == i


class TestCrossValidatePartition(PartitionerTestCase):
    def test_work(self):
        nb_samples = 20
        nb_blocks = 10
        in_blocks = [2, 3]
        it = CrossValidatePartitioner(nb_blocks, in_blocks).partition(
            self.get_data_column(nb_samples))
        data = list(it)
        assert data == [4, 5, 6, 7]


class TestTrain80Partitioner(PartitionerTestCase):
    def test_train(self):
        nb_samples = 21
        it = Train80Partitioner(True).partition(
            self.get_data_column(nb_samples))
        assert list(it) == list(range(16))

    def test_test(self):
        nb_samples = 21
        it = Train80Partitioner(False).partition(
            self.get_data_column(nb_samples))
        assert list(it) == list(range(16, 20))
