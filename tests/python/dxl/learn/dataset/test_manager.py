import unittest
from dxl.learn.dataset import DatasetManager

class TestDatasetManager(unittest.TestCase):
    def test_path_datasets(self):
        self.assertEqual(DatasetManager.path_datasets, '/something/wrong?')

