from dxl.learn.core import Model
from dxl.learn.function import flatten, ReLU, identity, OneHot, DropOut
from dxl.learn.model import DenseV2 as Dense
from dxl.learn.model import StackV2 as Stack
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
        for i in range(3):
            models += [Dense(self.config(self.KEYS.CONFIG.NB_UNITS)
                             [i], info='dense_{}'.format(i)),
                       ReLU,
                       DropOut()]
        # models.append(DropOut())
        models.append(
            Dense(self.config(self.KEYS.CONFIG.MAX_NB_HITS), info='dense_end'))
        seq = self.graphs.get('seq', Stack(info='stack', models=models))
        return seq(x)


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


import click
import os
from dxl.learn.core import Session


# @click.command()
# def test_model():
#     path_db = os.environ['GHOME'] + \
#         '/Workspace/IncidentEstimation/data/gamma.db'
#     padding_size = 5
#     d = create_dataset(dataset_db, path_db, padding_size, 32)
#     print(d)
#     model = IndexFirstHit('model', d.hits, padding_size, [100] * 3)
#     infer = model()

#     l = tf.losses.softmax_cross_entropy(d.first_hit_index.data, infer.data)
#     print(infer)
#     print(l)
#     acc, acc_op = tf.metrics.accuracy(tf.argmax(d.first_hit_index.data, 1),
#                                       tf.argmax(infer.data, 1))

#     t = Trainer('trainer', RMSPropOptimizer('opt', learning_rate=1e-3))
#     t.make({'objective': l})
#     train_step = t.train_step
#     with Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(tf.local_variables_initializer())
#         for i in range(10000):
#             sess.run(train_step)
#             if i % 100 == 0:
#                 print(sess.run([acc, acc_op]))
#                 print(sess.run([l, acc]))
