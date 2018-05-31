import unittest
from dxl.learn.test import TestCase
from dxl.learn.dataset import HDF5Loader


class TestHDF5Loader(TestCase):
    def get_test_hdf5_data_spec(self):
        return {
            'file_name': 'hdf5_loader_test.h5',
            'capacity': 100,
            'index': {
                1: [20, 20],
                2: [30, 30],
                3: [10, 10]
            },
        }

    def create_loader_for_test_hdf5_dataset(self):
        config = {
            'path_file':
            self.test_resource_path() /
            self.get_test_hdf5_data_spec()['file_name']
        }
        return HDF5Loader('dataset/test', config=config)

    def test_capacity(self):
        dataset = self.create_loader_for_test_hdf5_dataset()
        assert dataset.capacity == self.get_test_hdf5_data_spec()['capacity']

    def test_index(self):
        dataset = self.create_loader_for_test_hdf5_dataset()
        for i, v in self.get_test_hdf5_data_spec()['index']:
            self.assertFloatArrayEqual(v, dataset[i],
                                       'Test of index {} failed.'.format(i))
