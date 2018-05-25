"""
DataLoader utilities.

Class DataLoader is for pre_processing and indexs providing.

pre_processing include：filter, exclude etc. Then output indexs.

example:

"""
import h5py
import tables as tb 
import numpy as np 
from typing import Dict, Iterable
from dxl.fs import Path 


class DataLoader:
    class ENGINE:
        PYTABELS = 0
        H5PY = 1 
        NUMPY = 2
        NESTDIR = 3
    
    def __init__(self,
                 name:Path,
                 engine=0,
                 config:Dict=None):
        if config == None:
            raise ValueError("config is not allowed None")

        if engine == self.ENGINE.PYTABELS:
            self.ld_engine = TablesEngine(name, config)
        elif engine == self.ENGINE.H5PY:
            self.ld_engine = H5pyEngine(name, config)
        elif engine == self.ENGINE.NUMPY:
            self.ld_engine = NumpyEngine(name, config)
        elif engine == self.ENGINE.NESTDIR:
            self.ld_engine = NestDirEngine(name, config)
        else:
            raise ValueError("Not implement loader engine={}".format(engine))

    def __call__(self, mapattr, index=None):
        return self.ld_engine(mapattr, index)


class TablesEngine:
    '''Pytables Loader Engine
    Argumets:
        `name`: A fs.Path of hdf5 file
        `config`: A Dict
            `field`:
                # key := mapAttr
                # value := attrPath
                feature like fields:
                    {
                        'x': '/data/x',
                        'y': '/data/y'
                    }
                train test like fields:
                    {
                        'x_train': '/data/train/x',
                        'y_train': '/data/train/y',
                        'x_test': '/data/test/x',
                        'y_test': '/data/test/y'
                    }
            `pre_processing`:
                # key := attrPath
                # value := {op: op_cfg}
                {
                    '/data/y': {
                        'filter': {
                            value__lt: [1, 0, 0, 0] #lt := less than
                        }
                    }
                }
    Return:
        A Dict['mapAttr', indexs] where idexs := Iterable[int]  
    '''
    class KEYS:
        FIELD = 'field'
        PRE_PROCESS = 'pre_processing'
    
    def __init__(self, name, config):
        self.name = name
        self.fields = config[self.KEYS.FIELD]
        self.pre_process = config.get(self.KEYS.PRE_PROCESS)

        self.dl_stage = {}
        self.handles = {}
        self.construct()

    def construct(self):
       with tb.open_file(self.name, mode='r') as h5:
            for k, v in self.fields.items():
                node_path = Path(v).f   # dirname
                # node_attr = Path(v).n   # basename
                node_hdl = h5.get_node(node_path)
                if self.pre_process == None:
                    ids = node_hdl.nrow
                    self.handles.update({
                        node_path: {
                            'handel': node_hdl,
                            'index': list(range(ids))
                        }
                    })
                    self.dl_stage.update({k : node_path})
                else:
                    pass

    def __call__(self, mapattr, index):
        return self.loader(mapattr, index)

    def loader(self, mapatter, index):
        dl = {}


    def pre_processing(self):
        pass 


class H5pyEngine:
    pass 


class NumpyEngine:
    pass


class NestDirEngine:
    pass

