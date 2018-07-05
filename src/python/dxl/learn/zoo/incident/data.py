from typing import NamedTuple

import numpy as np
import tensorflow as tf

from dxl.data.function import (Filter, GetAttr, MapByPosition,
                               MapWithUnpackArgsKwargs, NestMapOf, OnIterator,
                               Padding, Swap, To, append, function, shape_list)
from dxl.data.zoo.incident_position_estimation.data import (CoincidenceColumns,
                                                            PhotonColumns,
                                                            ShuffledHitsColumns,
                                                            ShuffledHitsTable)
from dxl.data.zoo.incident_position_estimation.function import (raw_columns2shuffled_hits_columns,
                                                                sort_hits_by_energy,
                                                                filter_by_nb_hits,
                                                                drop_padded_hits)
from dxl.learn.core import Tensor
from dxl.learn.dataset import DatasetFromColumnsV2, Train80Partitioner
from dxl.learn.function import OneHot


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
def dataset_db(path_db, limit, is_coincidence, padding_size, batch_size, is_shuffle):
    if is_coincidence:
        source_columns = CoincidenceColumns(path_db, True, limit, 100000)
    else:
        source_columns = PhotonColumns(path_db, True, limit, 100000)
    columns = raw_columns2shuffled_hits_columns(
        source_columns, padding_size, sort_hits_by_energy)
    dataset = DatasetFromColumnsV2('dataset', columns,
                                   batch_size=batch_size, is_shuffle=is_shuffle)
    dataset.make()
    return post_processing(dataset, padding_size)


@function
def dataset_pytable(path_table, batch_size, is_shuffle, nb_hits, is_train=None):
    table = ShuffledHitsTable(path_table)
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
