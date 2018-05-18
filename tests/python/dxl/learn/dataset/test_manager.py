import unittest
from dxl.learn.dataset import DatasetManager
from typing import Dict

class TestDatasetManager(unittest.TestCase):
    def test_path_datasets(self):
        self.assertEqual(DatasetManager.path_datasets, '/something/wrong?')

    def test_info_datasets(self):
        '''nb_classes, batch, shape...
        '''
        self.assertEqual(DatasetManager.info, 'something')
    
    def test_cache_datasets(self):
        '''cache all in memory
        '''
        self.assertEqual(DatasetManager.cache, 'success')
    
    def test_filter_datasets(self):
        '''filter shape, label
            or filter from a func
        '''
        self.assertEqual(DatasetManager.filter, 'ok')

    def test_reshape_datasets(self):
        '''crop, upsampling...
        '''
        self.assertEqual(DatasetManager.reshape, 'done')

    def test_batch_datasets(self):
        self.assertEqual(DatasetManager.batch, 'done')

    def test_save_datasets(self):
        '''save as a new datasets
        '''
        self.assertEqual(DatasetManager.save, 'done')
    
    def test_repeat_datasets(self):
        self.assertEqual(DatasetManager.repeat, 'done')
    
    def test_shard_datasets(self):
        self.assertEqual(DatasetManager.shard, 'done')

    def test_shuffle_datasets(self):
        self.assertEqual(DatasetManager.shuffle, 'done')
    

   