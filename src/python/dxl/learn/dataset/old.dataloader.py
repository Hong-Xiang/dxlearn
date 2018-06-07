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


class DLEngine:
    PYTABLES = 0
    H5PY = 1
    NUMPY = 2
    NESTDIR = 3


class DataLoader:
    def __init__(self,
                 name:Path,
                 engine=0,
                 config:Dict=None):
        if config == None:
            raise ValueError("config is not allowed None")

        self.name = name
        self.cfg = config
        self.ld_engine = self.loadengine(engine)
       
    def loadengine(self, engine):
        if engine == DLEngine.PYTABLES:
            ld_engine = TablesEngine(self.name, self.cfg)
        elif engine == DLEngine.H5PY:
            ld_engine =  H5pyEngine(self.name, self.cfg)
        elif engine == DLEngine.NUMPY:
            ld_engine = NumpyEngine(self.name, self.cfg)
        elif engine == DLEngine.NESTDIR:
            ld_engine = NestDirEngine(self.name, self.cfg)
        else:
            raise ValueError("Not implement loader engine={}".format(engine))
        return ld_engine

    def loader(self, key):
        return LoaderKernel(key, self.ld_engine)


class LoaderKernel:
    def __init__(self, key, ld_engine):
        self.k_name = key
        self.k_engine = ld_engine
        self.k_indexs, self.k_attrs = ld_engine(key)

    @property
    def capacity(self):
        return len(self.k_indexs)

    @property
    def name(self):
        return self.k_attrs

    def __getitem__(self, id):
        return self.k_engine.loader(self.k_name, id)


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
                # key := mapAttr
                # value := {op: op_cfg}
                {
                    'x_train': {
                        'filter': {
                            'value__lt': [1, 0, 0, 0] #lt := less than
                        }
                    }
                }
    Return:
        A Dict['mapAttr', indexs] where idexs := Iterable[int]  
    '''
    class KEYS:
        class CONFIG:
            FIELD = 'field'
            PRE_PROCESS = 'pre_processing'
            HANDEL = 'handel'
            INDEX = 'index'
            NODEATTR = 'nodeattr'
            MAPINDEX = 'mapindex'
        class PROCESS:
            EXCLUDE = 'exclude'
            FILTER = 'filter'
      
    def __init__(self, name, config):
        self.name = name
        self.fields = config[self.KEYS.CONFIG.FIELD]
        self.process_cfg = config.get(self.KEYS.CONFIG.PRE_PROCESS)

        self.nodes = {}
        self.nodeattr = {}
        self.mapattr = {}
        self.h5 = tb.open_file(self.name, mode='r')

        self.construct()

    def __del__(self):
        self.h5.close()

    def construct(self):
        nodepath = {}
        for k, v in self.fields.items():
            node_path = Path(v).f   
            node_name = Path(node_path).n
            org_attr = Path(v).n 
            nodepath.update({node_name: node_path})
            self.mapattr.update({k: org_attr})
            if self.nodeattr.get(node_name) == None:
                self.nodeattr.update({node_name: [k]})
            else:
                self.nodeattr[node_name].append(k)

        for name, path in nodepath.items():
            hdl = self.h5.get_node(path)
            map_flag = False
            if self.process_cfg == None:
                ids = list(range(hdl.nrows))
            else:
                for attr in self.nodeattr[name]:    
                    if attr in self.process_cfg.keys():
                        cfg = self.process_cfg[attr]
                        org_attr = self.mapattr[attr]
                        ids = self.pre_processing(hdl, org_attr, cfg)
                        map_flag = True
                    else:
                        ids = list(range(hdl.nrows))

            mapids = []
            if map_flag:
                for i, _ in enumerate(ids):
                    mapids.append(i)
            else:
                mapids = ids
    
            self.nodes.update({
                name: {
                    self.KEYS.CONFIG.HANDEL: hdl,
                    self.KEYS.CONFIG.MAPINDEX: mapids,
                    self.KEYS.CONFIG.INDEX: ids
                }
            })
                
    def __call__(self, key):
        index = self.nodes[key][self.KEYS.CONFIG.MAPINDEX]
        attrs = self.nodeattr[key]
        return index, attrs

    def loader(self, node_name, id):
        hdl = self.nodes[node_name][self.KEYS.CONFIG.HANDEL]
        mapindex = self.nodes[node_name][self.KEYS.CONFIG.MAPINDEX]
        index = self.nodes[node_name][self.KEYS.CONFIG.INDEX]
        trid = index[mapindex[id]]    
        return hdl[trid]

    def pre_processing(self, hdl, k, cfg):
        index = []
        for op, dep in cfg.items():
            if op == self.KEYS.PROCESS.EXCLUDE:
                for i, row in enumerate(hdl):
                    if row[k] not in dep:
                        index.append(i)
            elif op == self.KEYS.PROCESS.FILTER:
                for i, row in enumerate(hdl):
                    if row[k] in dep:
                        index.append(i)
            else:
                raise ValueError("Unknown pre_procesing label {}".format(op))
                
        return index

class H5pyEngine:
    '''H5py Loader Engine
    Argumets:
    '''
    class KEYS:
        FIELD = 'field'
        PRE_PROCESS = 'pre_processing'
        HANDEL = 'handel'
        INDEX = 'index'
        NODEATTR = 'nodeattr'
        MAPINDEX = 'mapindex'
    
    def __init__(self, name, config):
        self.h5 = h5py.File(name, "r")

    def __del__(self):
        self.h5.close()

    def construct(self):
        pass 

    def __call__(self, key):
        pass 

    def loader(self, node_name, id):
        pass 

    def pre_processing(self, hdl, k, cfg):
        raise NotImplementedError


class NumpyEngine:
    pass


class NestDirEngine:
    pass

