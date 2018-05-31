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
    class KEYS:
        GROUP = 'data'
        TRAIN = 'train'
        TEST = 'test'
        VALID = 'valid'
        TRAINID = 80
        TESTID = 20

    def create_tables(self, name):
        with tb.open_file(name, "w") as h5:
            grp = h5.create_group('/', self.KEYS.GROUP)
            train_tb = h5.create_table(grp, self.KEYS.TRAIN, TbDesp)
            test_tb = h5.create_table(grp, self.KEYS.TEST, TbDesp)
            train_row = train_tb.row 
            test_row = test_tb.row
            for i in range(self.KEYS.TRAINID):
                train_row['x'] = 1 + np.ones([32, 32, 1], np.int8)
                label = np.zeros(10, np.int8)
                label[i%10] = 1
                train_row['y'] = label
                train_row.append()
            for i in range(self.KEYS.TESTID):
                test_row['x'] = 1 + np.ones([32, 32, 1], np.int8)
                label = np.zeros(10, np.int8)
                label[i%10] = 1
                test_row['y'] = label
                test_row.append()
            train_tb.flush()
            test_tb.flush()
    
    def test_LoaderKernel(self):
        name = os.path.join(DATATEST, "mnist_test.h5")
        self.create_tables(name)
        engine = DLEngine.PYTABLES
        config = {
            'field':{
                'x_train': '/data/train/x',
                'y_train': '/data/train/y',
                'x_test': '/data/test/x',
                'y_test': '/data/test/y'
            }
        }
        _attrs = ['x_train', 'y_train']
        _capacity = 80
        datald = DataLoader(name, engine, config)
        train_ld = datald.loader('train')
        assert isinstance(train_ld, LoaderKernel)
        attrs = train_ld.name
        capacity = train_ld.capacity
        self.assertEqual(attrs, _attrs)
        self.assertEqual(capacity, _capacity)

    def test_TablesEngine_preprocess(self):
        name = os.path.join(DATATEST, "mnist_test.h5")
        self.create_tables(name)
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
                    'exclude': [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],], # get 1~9
                },
                'y_test': {
                    'exclude': [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],], # get 1~9
                }
            }
        }
        datald = DataLoader(name, engine, config)
        _capacity = 18
        test_ld = datald.loader('test')
        capacity = test_ld.capacity
        self.assertEqual(capacity, _capacity)
    
    def test_TablesEngine_loader(self):
        name = os.path.join(DATATEST, "mnist_test.h5")
        self.create_tables(name)
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
                    'exclude': [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],], # get 1~9
                },
                'y_test': {
                    'exclude': [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],], # get 1~9
                }
            }
        }
        datald = DataLoader(name, engine, config)
        test_ld = datald.loader('test')
        id = 9
        _value = (id + 2) + np.ones([32, 32, 1], np.int8)
        self.assertEqual(test_ld[id]['x_test'], _value)
    
# class TestDataLoaderKernel(unittest.TestCase):
#     def set_up_fixture(self):
#         pass
#     def test_TablesEngine(self):
#         name = os.path.join(HOME, 'testdata/dataset/mnist_test.h5')
#         engine = DLEngine.PYTABLES
#         config = {
#             'field':{
#                 'x_train': '/data/train/x',
#                 'y_train': '/data/train/y',
#                 'x_test': '/data/test/x',
#                 'y_test': '/data/test/y'
#             },
#             'pre_processing':{
#                 'y_train': {
#                     'exclude': {
#                         'value': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # get 1~9
#                     }
#                 },
#                 'y_test': {
#                     'exlude': {
#                         'value': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # get 1~9
#                     }
#                 }
#             }
#         }
#         datald = DataLoader(name, engine, config)
#         datald_kernel = datald.loader('train'), 100, range(100)
        

        
