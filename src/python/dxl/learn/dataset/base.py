import tables as tb
from ..core import Graph
from dxl.fs import Path
from typing import Dict
from .partition import Partition


class Dataset(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR:
            pass
        class CONFIG:
            pass 

    def __init__(self,
                 name,
                 tensors=None,
                 subgraphs=None,
                 info=None,
                 config=None):
        super.__init__(name, tensors, subgraphs, info, config)
        self.construct(name)

    def construct(self, name):
        self.handel = self.loader(name)
        self.pre_processing(self.handel)
        self.partition()
        self.post_processing()
   
    def loader(self, name):
        return NotImplementedError

    def pre_processing(self, handel):
        pass 
    
    def capacity(self):
        return NotImplementedError

    def partition(self):
        pass

    def post_processing(self):
        pass 
        # x = tf.data.Dataset(...)
        # x = tf.data.make_one_shot_iterator()
        # x = x.next()
        # self.tensors['x'] = ...
   

class HDF5Dataset(Dataset):
    '''Default pytables
    '''
    class KEYS(Dataset.KEYS):
        class TENSOR:
            pass 
        class CONFIG:
            IN_MEMORY = 'in_memory'
            FIELD = 'field'
        class CMD:
            ITER = 'hand=h5.{}.iterrows'
        
    def __init__(self, name, config, info=None):
        super().__init__(
            name=name,
            config=config,
            info=info)
    
    def loader(self, name):
        with tb.open_file(name, mode="r") as h5:
            handels = {}
            field = self.config(self.KEYS.CONFIG.FIELD)
            for k, v in field.items():
                hand = []
                cmd = 'hand=h5.{}.iterrows'.format(v)
                exec(cmd)
                
                if self.config(self.KEYS.CONFIG.IN_MEMORY):
                    hand = []
                    cmd = 'hand=[x[{}] for x in h5.{}.iterow]' 
                else:
                   
                handels.update({k : hand})

            return handels

    def pre_processing(self, handel):
        pass
    
           


class FileDataset(Dataset):
    pass 


class NpyDataset(Dataset):
    pass

