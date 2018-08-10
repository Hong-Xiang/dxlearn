from typing import NamedTuple

import numpy as np
import tensorflow as tf

from dxl.data.function import (Filter, GetAttr, MapByPosition,
                               MapWithUnpackArgsKwargs, NestMapOf, OnIterator,
                               Padding, Swap, To, append, function, shape_list, Function)
from dxl.learn.core import Tensor
from dxl.learn.dataset import DatasetFromColumnsV2, Train80Partitioner
from dxl.learn.function import OneHot

from dxl.data.zoo.incident import (
    load_table, Photon, Coincidence, PhotonColumns, CoincidenceColumns)

from dxl.learn.function import Sum, ArgMax
from functools import singledispatch

from random import shuffle


class SplitByPeriod(Function):
    def __init__(self, period, fst_indices):
        self.period = period
        self.fst_indices = fst_indices

    def __call__(self, arr):
        fst = [x for i, x in enumerate(arr)
               if (i % self.period) in self.fst_indices]
        snd = [x for i, x in enumerate(arr)
               if not (i % self.period) in self.fst_indices]
        return fst, snd


def reindex_crystal(x: Photon):
    if x.hits[0].crystal_index is None:
        return x
    crystal_indices = [h.crystal_index for h in x.hits]
    crystal_indices = list(set(crystal_indices))
    crystal_indices.sort()
    new_indices = [crystal_indices.index(h.crystal_index) for h in x.hits]
    hits = [h.update(crystal_index=i) for h, i in zip(x.hits, new_indices)]
    return x.update(hits=hits)


def binary_crystal_index(p: Photon):
    first_hit_crystal_id = p.hits[p.first_hit_index].crystal_index
    new_indices = [1 if h.crystal_index == first_hit_crystal_id else 0
                   for h in p.hits]
    hits = [h.update(crystal_index=i) for h, i in zip(p.hits, new_indices)]
    return p.update(hits=hits)


class FilterPhotonByNbTrueHits(Function):
    def __init__(self, nb_hits):
        self.nb_hits = nb_hits

    def __call__(self, photons):
        return [p for p in photons if p.nb_ture_hits == self.nb_hits]


def parse_hits_features(p):
    return [[h.x, h.y, h.z, h.e] for h in p.hits]


def parse_crystal_index(p):
    return [h.crystal_index for h in p.hits]




class DatasetIncidentSingle(NamedTuple):
    hits: Tensor
    first_hit_index: Tensor
    padded_size: Tensor


@function
def post_processing(dataset, padding_size):
    hits = dataset.tensors['hits']
    shape = shape_list(hits.data)
    shape[1] = padding_size

    hits = Tensor(tf.reshape(hits.data, shape))
    label = Tensor(OneHot(padding_size)(
        dataset.tensors['first_hit_index'].data))
    return DatasetIncidentSingle(hits, label, dataset.tensors['padded_size'])


@function
def photon_pytable(path_table, batch_size, is_shuffle, nb_hits, is_train=None):
    photons = load_table(path_table)

    padding_size = table.padding_size
    table = filter_by_nb_hits(table, nb_hits)
    table = drop_padded_hits(table, nb_hits)
    if is_train is not None:
        table = ShuffledHitsColumns(table.dataclass,
                                    list(Train80Partitioner(is_train).partition(table)))
    dataset = DatasetFromColumnsV2('dataset',
                                   table,
                                   batch_size=batch_size, is_shuffle=is_shuffle)
    dataset.make()
    return post_processing(dataset, nb_hits)


@function
def dataset_fast(path_table, batch_size, is_shuffle, nb_hits, is_train):
    table = ShuffledHitsTable(path_table)
    padding_size = table.padding_size
    table = filter_by_nb_hits(table, nb_hits)
    # table = drop_padded_hits(table, nb_hits)
    if is_train is not None:
        table = ShuffledHitsColumns(table.dataclass,
                                    list(Train80Partitioner(is_train).partition(table)))
    dtypes = {
        'hits': np.float32,
        'first_hit_index': np.int32,
        'padded_size': np.int32,
    }
    data = {k: np.array([getattr(table.data[i], k) for i in range(table.capacity)],
                        dtype=dtypes[k])
            for k in table.columns}
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.repeat()
    if is_shuffle:
        dataset = dataset.shuffle(4 * batch_size)
    dataset = dataset.batch(batch_size)
    tensors = dataset.make_one_shot_iterator().get_next()
    for k in tensors:
        tensors[k] = Tensor(tensors[k])
    tensors['first_hit_index'] = OneHot(
        tensors['hits'].shape[1])(tensors['first_hit_index'])
    return DatasetIncidentSingle(tensors['hits'], tensors['first_hit_index'], tensors['padded_size'])


__all__ = ['dataset_db', 'dataset_pytable',
           'DatasetIncidentSingle', 'dataset_fast']
# if __name__ == "__main__":
#     padding_size = 10
#     batch_size = 32
#     path_db = 'data/gamma.db'
#     d_tuple = create_dataset(dataset_db, path_db, padding_size, batch_size)
#     print(NestMapOf(GetAttr('shape'))(d_tuple))
#     nb_batches = 100
#     samples = []
#     with tf.Session() as sess:
#         for i in range(nb_batches):
#             samples.append(sess.run(NestMapOf(GetAttr('data'))(d_tuple)))
#     hits = np.array([s.hits for s in samples])
#     first_index = np.array([s.first_hit_index for s in samples])
#     padded_size = np.array([s.padded_size for s in samples])

# np.savez('fast_data.npz', hits=hits, first_index=first_index, padded_size=padded_size)
