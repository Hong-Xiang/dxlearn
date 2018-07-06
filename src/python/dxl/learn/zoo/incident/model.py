from dxl.learn.core import Model
from dxl.learn.function import flatten, ReLU, identity, OneHot, DropOut
from dxl.learn.model import DenseV2 as Dense
from dxl.learn.model import Sequential
from .data import DatasetIncidentSingle, dataset_db, dataset_pytable
import numpy as np
import tensorflow as tf
from dxl.learn.network.trainer import Trainer, RMSPropOptimizer
from dxl.data.function import shape_list


class IndexFirstHit(Model):
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
        return self.graphs(self.KEYS.GRAPH.SEQUENTIAL)(x)


def placeholder_input():
    return DatasetIncidentSingle(
        hits=tf.placeholder(tf.float32, [32, 10, 4]),
        first_hit_index=tf.placeholder(tf.int32, [32]),
        padded_size=tf.placeholder(tf.int32, [32])
    )


def dummy_input():
    return DatasetIncidentSingle(
        hits=np.ones([32, 10, 4]),
        first_hit_index=np.random.randint(0, 9, [32]),
        padded_size=np.random.randint(0, 9, [32]),
    )
