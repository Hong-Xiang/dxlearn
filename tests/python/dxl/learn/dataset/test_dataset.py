import os
import unittest
from dxl.learn.dataset import Dataset

HOME = os.environ['HOME']

class TestDataset(unittest.TestCase):
    def test_descrip(self):
        data_path = os.path.join(HOME, 'testdata/dataset/mnist.h5')
        config = {
            'cache': True,
            'batch': None,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l'
            }
        }
        _descrip = {
            'name': 'mnist.h5',
            'x_l': {
                'shape': (32, 32, 3),
                'table': 200,
                'dtype': 'UInt16Col'
            },
            'y_l': {
                'shape': (1, 10),
                'table': 200,
                'dtype': 'UInt8Col'
            }
        }
        mnist_data = Dataset(
            name=data_path,
            graph_info=None,
            config=config)
        descrip = mnist_data.descrip
        self.assertDictEqual(descrip, _descrip)
    
    def test_cache_datasets(self):
        data_path = os.path.join(HOME, 'testdata/dataset/mnist.h5')
        nocache_cfg = {
            'cache': False,
            'batch': None,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l'
            }
        }
        mnist_nocache = Dataset(
            name=data_path,
            graph_info=graph_info,
            config=nocache_cfg)
        time0 = mnist_nocache.time_cost()

        cache_cfg = {
            'cache': True,
            'batch': None,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l'
            }
        }
        mnist_cache = Dataset(
            name=data_path,
            graph_info=None,
            config=cache_cfg)
        time1 = mnist_cache.time_cost()
        self.assertFalse(int(time1)-int(time0))
    
    def test_filter_datasets(self):
        data_path = os.path.join(HOME, 'testdata/dataset/mnist.h5')
        config = {
            'cache': True,
            'batch': None,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l'
            },
            'processing':{
                'y_l': [
                    'filter': {
                        'value__lt': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],    #lt := less than
                        'relate': ('x_l',)    #relate: filter x_l relate to y_l
                    }
                ]
            }
        }
        _descrip = {
            'name': 'mnist.h5',
            'x_l': {
                'shape': (32, 32, 3),
                'table': 180,
                'dtype': 'UInt16Col'
            },
            'y_l': {
                'shape': (1, 10),
                'table': 180,
                'dtype': 'UInt8Col'
            }
        }
        mnist = Dataset(
            name=data_path,
            graph_info=None,
            config=config)
        descrip = mnist.descrip
        self.assertDictEqual(_descrip, descrip)
        
    def test_exclude_datasets(self):
        data_path = os.path.join(HOME, 'testdata/dataset/mnist.h5')
        config = {
            'cache': True,
            'batch': None,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l'
            },
            'processing':{
                'y_l': [
                    'exclude': {    
                        'value': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # value == 
                        'relate': ('x_l',)    #relate: filter x_l relate to y_l
                    }
                ]
            }
        }
        _descrip = {
            'name': 'mnist.h5',
            'x_l': {
                'shape': (32, 32, 3),
                'table': 180,
                'dtype': 'UInt16Col'
            },
            'y_l': {
                'shape': (1, 10),
                'table': 180,
                'dtype': 'UInt8Col'
            }
        }
        mnist = Dataset(
            name=data_path,
            graph_info=None,
            config=config)
        descrip = mnist.descrip
        self.assertDictEqual(_descrip, descrip)

    def test_reshape_datasets(self):
        data_path = os.path.join(HOME, 'testdata/dataset/mnist.h5')
        config = {
            'cache': True,
            'batch': None,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l'
            },
            'processing':{
                'x_l': [
                    'reshape': {    
                        'target': (24, 24, 3), 
                        'method': 0
                    }
                ]
            }
        }
        _descrip = {
            'name': 'mnist.h5',
            'x_l': {
                'shape': (24, 24, 3),
                'table': 200,
                'dtype': 'UInt16Col'
            },
            'y_l': {
                'shape': (1, 10),
                'table': 200,
                'dtype': 'UInt8Col'
            }
        }
        mnist = Dataset(
            name=data_path,
            graph_info=None,
            config=config)
        descrip = mnist.descrip
        self.assertDictEqual(_descrip, descrip)

    def test_batch_datasets(self):
        data_path = os.path.join(HOME, 'testdata/dataset/mnist.h5')
        config = {
            'cache': True,
            'batch': 8,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l'
            }
        }
        _descrip = {
            'name': 'mnist.h5',
            'x_l': {
                'shape': (8, 24, 24, 3),
                'table': 25,
                'dtype': 'UInt16Col'
            },
            'y_l': {
                'shape': (8, 1, 10),
                'table': 25,
                'dtype': 'UInt8Col'
            }
        }
        mnist = Dataset(
            name=data_path,
            graph_info=None,
            config=config)
        descrip = mnist.descrip
        self.assertDictEqual(_descrip, descrip)

    def test_save_datasets(self):
        data_path = os.path.join(HOME, 'testdata/dataset/mnist.h5')
        config = {
            'cache': True,
            'batch': 8,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l',
                'x_s': 'train/small/x_s',
                'y_s': 'train/small/y_s'
            },
            'processing': {
                'x_l': [
                    'reshape': {    
                        'target': (24, 24, 3), 
                        'method': 0
                    }
                ]
            },
            'save': {
                'file_path': os.path.join('HOME', 'testdata/dataset/mnist_cb8.h5'),
                'field': {
                    'x': ['train/x', ('x_l', 'x_s')],
                    'y': ['train/y', ('y_l', 'y_s')]
                }
            }
        }
        _descrip = {
            'name': 'mnist_cb8.h5',
            'x': {
                'shape': (8, 24, 24, 3),
                'table': 50,
                'dtype': 'UInt16Col'
            },
            'y': {
                'shape': (8, 1, 10),
                'table': 50,
                'dtype': 'UInt8Col'
            }
        }
        mnist = Dataset(
            name=data_path,
            graph_info=None,
            config=config)
        descrip = mnist.save().descrip
        self.assertDictEqual(_descrip, descrip)
    
    def test_repeat_datasets(self):
        config = {
            'cache': True,
            'batch': 8,
            'repeat': 1,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l'
            }
        }
        _descrip = {
            'name': 'mnist.h5',
            'x_l': {
                'shape': (8, 32, 32, 3),
                'table': 50,
                'dtype': 'UInt16Col'
            },
            'y_l': {
                'shape': (8, 1, 10),
                'table': 50,
                'dtype': 'UInt8Col'
            }
        }
        mnist = Dataset(
            name=data_path,
            graph_info=None,
            config=config)
        descrip = mnist.descrip
        self.assertDictEqual(_descrip, descrip)

    def test_shuffle_datasets(self):
        config = {
            'cache': False,
            'batch': 8,
            'shuffle': 20,
            'field': {
                'x_l': 'train/large/x_l',
                'y_l': 'train/large/y_l'
            }
        }
        _descrip = {
            'name': 'mnist.h5',
            'x_l': {
                'shape': (8, 32, 32, 3),
                'table': 25,
                'dtype': 'UInt16Col'
            },
            'y_l': {
                'shape': (8, 1, 10),
                'table': 25,
                'dtype': 'UInt8Col'
            },
            'shuffle': 20
        }
        mnist = Dataset(
            name=data_path,
            graph_info=None,
            config=config)
        descrip = mnist.descrip
        self.assertDictEqual(_descrip, descrip)
