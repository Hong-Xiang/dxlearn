from dxl.data.zoo.incident_position_estimation import (
    Hit, HitWithCrystalCenter, PhotonColumns, sort_hits_by_energy, ShuffledHitsWithIndexAndPaddedSize, ShuffledHitsColumns, photon2hits)
from dxl.data.function import GetAttr, MapByPosition, MapWithUnpackArgsKwargs, append, Swap, To, Padding, OnIterator, NestMapOf, shape_list, function
from dxl.learn.dataset import DatasetFromColumnsV2
from dxl.learn.function import OneHot
import tensorflow as tf
import numpy as np
from typing import NamedTuple
from dxl.learn.core import Tensor
from dxl.data.zoo.incident_position_estimation.columns import HitsColumnFromNPZ
from dxl.data.zoo.incident_position_estimation import photon2hits
from dxl.data.function import Filter


@function
def dataset_db(path_db, padding_size, batch_size, is_shuffle):
    columns = ShuffledHitsColumns(PhotonColumns(path_db, HitWithCrystalCenter),
                                  ShuffledHitsWithIndexAndPaddedSize,
                                  OnIterator(photon2hits(
                                      sort_hits_by_energy, padding_size))
                                  >> Filter(lambda x: x.hits.shape[0] <= padding_size))
    dataset = DatasetFromColumnsV2('dataset', columns,
                                   batch_size=batch_size, is_shuffle=is_shuffle)
    return dataset


@function
def dataset_npz(path_npz, padding_size, batch_size, is_shuffle):
    if padding_size != 10:
        raise ValueError("Padding size is fixed to 10 for NPZ dataset.")
    columns = HitsColumnFromNPZ(path_npz)
    dataset = DatasetFromColumnsV2(
        'dataset', columns, batch_size=batch_size, is_shuffle=is_shuffle)
    return dataset


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


def create_dataset(dataset_maker, path_source, padding_size, batch_size):
    d = dataset_maker(path_source, padding_size, batch_size, True)
    d.make()
    d_tuple = post_processing(d, padding_size)
    return d_tuple


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
