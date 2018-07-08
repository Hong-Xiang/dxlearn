from dxl.learn.core import Model
from dxl.learn.function import flatten, ReLU, identity, OneHot, DropOut
from dxl.learn.model import DenseV2 as Dense
from dxl.learn.model import Sequential
import numpy as np
import tensorflow as tf


class FirstHit(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            HITS = 'hits'

        class CONFIG(Model.KEYS.CONFIG):
            MAX_NB_HITS = 'max_nb_hits'
            NB_UNITS = 'nb_units'

        class GRAPH(Model.KEYS.GRAPH):
            SEQUENTIAL = 'sequential'

    def __init__(self, info, hits=None, max_nb_hits=None, nb_units=None):
        super().__init__(info, tensors={self.KEYS.TENSOR.HITS: hits},
                         config={
            self.KEYS.CONFIG.MAX_NB_HITS: max_nb_hits,
            self.KEYS.CONFIG.NB_UNITS: nb_units
        })

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.HITS]
        x = flatten(x)
        m = identity
        models = []
        for i in range(len(self.config(self.KEYS.CONFIG.NB_UNITS))):
            models += [Dense(self.config(self.KEYS.CONFIG.NB_UNITS)
                             [i], info='dense_{}'.format(i)),
                       ReLU,
                       DropOut()]
        models.append(
            Dense(self.config(self.KEYS.CONFIG.MAX_NB_HITS), info='dense_end'))
        if self.graphs.get(self.KEYS.GRAPH.SEQUENTIAL) is None:
            self.graphs[self.KEYS.GRAPH.SEQUENTIAL] = Sequential(
                info='stack', models=models)
        return self.graphs[self.KEYS.GRAPH.SEQUENTIAL](x)
