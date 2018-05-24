import dxl.learn.dataset as dld
import tensorflow as tf

class TestDataset(tf.test.TestCase):
    def create_partition(self):
        pass
    def test_construct(self):
        d = dld.Dataset()
