import unittest
# from dxl.learn.dataset import DatasetManager
from typing import Dict
import pytest

@pytest.mark.skip(reason='not impl yet')
class TestDatasetManager(unittest.TestCase):
    def test_path_datasets(self):
        self.assertEqual(DatasetManager.path_datasets, '/something/wrong?')
        
    

   