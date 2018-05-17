import dxl.learn.dataset as dld
import unittest


class TestTrain80Partition(unittest.TestCase):
    def test_ids(self):
        spec = {
            'batch_size': 32,
            'dataset_size': 100,
        }
        p = dld.Train80Partition(**spec)