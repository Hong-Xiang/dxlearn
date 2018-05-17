from dxl.learn.dataset import HDF5Dataset
import tensorflow as tf

def create_batch_partition(batch_size):
    class BatchStub:
        pass
    return BatchStub

class TestHDF5Dataset(tf.test.TestCase):
    def test_construct(self):
        spec = {
            'name': 'mnist'
            'path_file': './testdata/dataset/mnist.h5'
            'path_datasets': {
                'x': 'train/x',
                'y': 'train/y'
            }
        }
        BATCH_SIZE = 32 
        partition = create_batch_partition(BATCH_SIZE)
        dataset = HDF5Dataset(**spec, partition=partition)
        assert dataset.tensor('x').shape == [100, 28, 28, 1]

