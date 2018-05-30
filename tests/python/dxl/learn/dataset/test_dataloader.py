import os 
import unittest
import h5py
import tables as tb
import numpy as np
from dxl.learn.dataset import DataLoader, LoaderKernel

HOME = os.environ['HOME']
DATATEST = os.path.join(HOME, "dataset", "test")

class DLEngine:
    PYTABLES = 0
    H5PY = 1
    NUMPY = 2
    NESTDIR = 3


class TbDesp(tb.IsDescription):
    x = tb.UInt16Col(shape=(32, 32, 1))
    y = tb.UInt16Col(shape=(10))


class TestDataLoader(unittest.TestCase):
    group = 'data'
    train = 'train'
    test = 'test'
    valid = 'valid'
    trainid = 80
    testid = 20

    def create_tables(self, name):
        with tb.open_file(name, "w") as h5:
            grp = h5.create_group('/', self.group)
            train_tb = h5.create_table(grp, self.train, TbDesp)
            test_tb = h5.create_table(grp, self.test, TbDesp)
            train_row = train_tb.row 
            test_row = test_tb.row
            for i in range(self.trainid):
                train_row['x'] = 1 + np.ones([32, 32, 1], np.int8)
                label = np.zeros(10, np.int8)
                label[i%10] = 1
                train_row['y'] = label
                train_row.append()
            for i in range(self.testid):
                test_row['x'] = 1 + np.ones([32, 32, 1], np.int8)
                label = np.zeros(10, np.int8)
                label[i%10] = 1
                test_row['y'] = label
                test_row.append()
            train_tb.flush()
            test_tb.flush()
    
    def create_h5py(self, name):
        pass
        # with h5py.File(name, "w") as f:
        #     grp = f.create_group(self.group)
        #     train_tb = grp.create_dataset(self.train, (80,))
        #     test_tb = grp.create_dataset(self.test, (20,))

    def test_TablesEngine(self):
        name = os.path.join(DATATEST, "mnist_test.h5")
        engine = DLEngine.PYTABLES
        config = {
            'field':{
                'x_train': '/data/train/x',
                'y_train': '/data/train/y',
                'x_test': '/data/test/x',
                'y_test': '/data/test/y'
            }
        }
        _attrs = config['field'].keys()
        datald = DataLoader(name, engine, config)
        assert isinstance(datald.loader('train'), LoaderKernel)
        mapattrs = datald.loader('train').name
        self.assertEqual(mapattrs, _attrs)
       

class TestDataLoaderKernel(unittest.TestCase):
    def set_up_fixture(self):
        pass
    def test_TablesEngine(self):
        name = os.path.join(HOME, 'testdata/dataset/mnist.h5')
        engine = DLEngine.PYTABLES
        config = {
            'field':{
                'x_train': '/data/train/x',
                'y_train': '/data/train/y',
                'x_test': '/data/test/x',
                'y_test': '/data/test/y'
            },
            'pre_processing':{
                'y_train': {
                    'exclude': {
                        'value': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # get 1~9
                    }
                },
                'y_test': {
                    'exlude': {
                        'value': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # get 1~9
                    }
                }
            }
        }
        datald = DataLoader(name, engine, config)
        datald_kernel = datald.loader('train'), 100, range(100)
        

        
