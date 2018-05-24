import os
import tensorflow as tf
from dxpy.configs import configurable, ConfigsView
import numpy as np
from dxpy.learn.graph import Graph, NodeKeys
from dxpy.learn.config import config

DEFAULT_FILE_NAME = 'mCT.npy'
NB_IMAGES = 983


def data_type_np(key):
    if key == 'id':
        return np.int64
    else:
        return np.float32


def data_type_tf(key):
    if key == 'id':
        return tf.int64
    else:
        return tf.float32


def data_shape(key):
    return {
        'phantom': [192, 192],
        'sinogram1x': [768 * 2, 384],
        'sinogram2x': [384 * 2, 192],
        'sinogram4x': [192 * 2, 96],
        'sinogram8x': [96 * 2, 48],
        'id': [],
    }[key]


@configurable(ConfigsView(config, 'datasets/mCT_sinograms'))
def _npy_file(path: str, npy_file_name=DEFAULT_FILE_NAME):
    from dxpy.core.path import Path
    return str(Path(path) / npy_file_name)


def _load_sample(idx, data):
    result = dict()
    result['phantom'] = data[idx][4]
    for i in range(4):
        result['sinogram{}x'.format(2**i)] = data[idx][i].T
    result['id'] = idx
    return result


def _processing(result):
    from dxpy.tensor.transform import rotate
    for k in result:
        if k == 'id':
            continue
        if not k == 'phantom':
            sino = result[k]
            sino_d = np.flip(sino, axis=0)
            sino_d = rotate(sino_d, -1, 0)
            result[k] = np.concatenate([sino, sino_d], axis=1)
            result[k] = result[k].T
            result[k] = np.concatenate([result[k]] * 2, axis=0)
        result[k] = result[k].astype(data_type_np(k))
        # if k == 'phantom':
            # result[k] = result[k] / np.sum(result[k]) * 1e6
        # if k == 'phantom':
            # result[k] = result[k] / np.sum(result[k]) * 4e6
    return result


def dataset_generator(ids):
    data = np.load(_npy_file())
    for idx in ids:
        result = _load_sample(idx, data)
        result = _processing(result)
        yield result


class Dataset(Graph):
    MEAN = 9.93 
    STD = 7.95
    BASE_SHAPE = [384, 384]

    @configurable(config, with_name=True)
    def __init__(self, name='mCT_sinograms', batch_size=32, ids=None, shuffle=True, dataset_type='train', **kw):
        super().__init__(name=name, batch_size=batch_size,
                         shuffle=shuffle, dataset_type=dataset_type, **kw)
        if ids is None:
            if dataset_type == 'train':
                self._ids = list(range(0, int(NB_IMAGES * 0.8)))
            elif dataset_type == 'test':
                self._ids = list(range(int(NB_IMAGES * 0.8), NB_IMAGES))
            elif dataset_type == 'full':
                self._ids = list(range(NB_IMAGES))
        else:
            self._ids = ids
        self._keys = ['id', 'phantom'] + \
            ['sinogram{}x'.format(2**i) for i in range(4)]

        dataset = self._create_dataset()
        dataset = self._processing_dataset(dataset)
        self._register_dataset(dataset)

    def _create_dataset(self):
        from functools import partial
        output_types = {k: data_type_tf(k) for k in self._keys}
        output_shapes = {k: data_shape(k) for k in self._keys}
        dataset_generator_partial = partial(dataset_generator, ids=self._ids)
        return tf.data.Dataset.from_generator(dataset_generator_partial,
                                              output_types,
                                              output_shapes)

    def _format_tensors(self, data):
        return {k: tf.reshape(tf.cast(data[k], data_type_tf(k)), data_shape(k)) for k in data}

    def _processing_dataset(self, dataset):
        dataset = (dataset
                   .repeat()
                   .map(self._format_tensors))
        if self.param('shuffle'):
            dataset = dataset.shuffle(1024)
        return dataset.batch(self.param('batch_size'))

    def _register_dataset(self, dataset):
        from ...utils.tensor import ensure_shape
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        # self.register_main_node(next_element)
        for k in self._keys:
            shape = [self.param('batch_size')] + list(data_shape(k)) + [1]
            self.register_node(k, ensure_shape(next_element[k], shape=shape))
