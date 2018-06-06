from dxl.learn.dataset.partition import Partition, CrossValidatePartition, Train80Partition
import unittest
import pytest


class TestPartition(unittest.TestCase):
    def test_next(self):
        nb_samples = 10
        p = Partition(range(nb_samples))
        nb_tests = 20
        for i in range(nb_tests):
            assert next(p) == i % nb_samples

    def test_iter(self):
        nb_samples = 10
        nb_epoch = 2
        p = Partition(range(nb_samples), nb_epoch)
        result = []
        sampled = 0
        for i in p:
            sampled += 1
            result.append(i)
            if sampled > nb_samples * nb_epoch:
                self.fail('Should stop when exceeds nb_epoch')
        expect = list(range(nb_samples)) * 2
        assert result == expect

    def test_capacity(self):
        nb_samples = 10
        nb_epoch = 2
        p = Partition(range(nb_samples), nb_epoch)
        assert p.capacity == nb_epoch * nb_samples


class TestCrossValidatePartition(unittest.TestCase):
    def test_work(self):
        nb_blocks = 10
        nb_samples = 20
        in_blocks = [2, 3]
        nb_epochs = 1
        p = CrossValidatePartition(range(nb_samples), nb_blocks, in_blocks)
        assert list(p.indices) == list(range(2 * 2, 2 * 4))


class TestTrain80Partition(unittest.TestCase):
    def test_train(self):
        nb_samples = 21
        p = Train80Partition(range(nb_samples), True)
        assert list(p.indices) == list(range(16))

    def test_test(self):
        nb_samples = 21
        p = Train80Partition(range(nb_samples), False)
        assert list(p.indices) == list(range(16, 20))
