from dxl.learn.dataset import DatasetFromColumns, RangeColumns, Train80Partitioner, DataColumnsPartition
from dxl.learn.test import TestCase
import pytest
import numpy as np


class TestDatasetFromColumns(TestCase):
    def get_dataset(self,
                    nb_samples=100,
                    nb_epochs=1,
                    batch_size=32,
                    is_shuffle=False):
        d = DatasetFromColumns(
            'datset',
            DataColumnsPartition(
                RangeColumns(nb_samples), Train80Partitioner(True)),
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            is_shuffle=is_shuffle)
        return d

    def test_construct(self):
        d = self.get_dataset()
        assert d.tensors[d.KEYS.TENSOR.DATA].shape == [32]

    def test_sample(self):
        d = self.get_dataset()
        samples = []
        nb_batch = 2
        with self.test_session() as sess:
            for i in range(nb_batch):
                samples.append(sess.run(d.tensors[d.KEYS.TENSOR.DATA]))
        samples = np.array(samples)
        expected = np.zeros([nb_batch, 32])
        for i in range(nb_batch):
            for j in range(32):
                expected[i, j] = i * 32 + j
        self.assertFloatArrayEqual(expected, samples, 'samples not equal')
