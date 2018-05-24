import os
import tensorflow as tf
from dxpy.configs import configurable, ConfigsView
import numpy as np
from dxpy.learn.graph import Graph, NodeKeys
from dxpy.learn.config import config

DEFAULT_FILE_NAME = 'pet_rebin2.h5'
NB_IMAGES = 4770
CROP_OFFSET = 32


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
        'sinogram': [640, 128 - 2 * CROP_OFFSET],
        'id': [],
    }[key]


@configurable(ConfigsView(config, 'datasets/mice_sinograms'))
def _h5_file(path: str, file_name=DEFAULT_FILE_NAME):
    from dxpy.core.path import Path
    return str(Path(path) / file_name)


def _load_sample(idx, data):
    result = {'id': idx, 'sinogram': data[idx, ...].T}
    return result


def _processing(result):
    import numpy as np
    result['sinogram'] = np.concatenate(
        [result['sinogram']] * 2)[:, CROP_OFFSET:-CROP_OFFSET]
    return result


def dataset_generator(ids):
    import h5py
    with h5py.File(str(_h5_file())) as fin:
        data = np.array(fin['sinograms'])
        data = np.maximum(data, 0.0)
    for idx in ids:
        result = _load_sample(idx, data)
        result = _processing(result)
        yield result


class Dataset(Graph):
    MEAN = 100.0
    STD = 150.0
    HIGH_NOISE_MAX_DEPTH = 128
    DEPTH_PER_MICE = 159

    @configurable(config, with_name=True)
    def __init__(self, name='mice_sinograms', batch_size=32, ids=None, shuffle=True, dataset_type='train', drop_end_sinos=True, **kw):
        super().__init__(name=name,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         dataset_type=dataset_type,
                         drop_end_sinos=drop_end_sinos,
                         **kw)
        if ids is None:
            if dataset_type == 'train':
                self._ids = list(range(0, int(NB_IMAGES * 0.8)))
            elif dataset_type == 'test':
                self._ids = list(range(int(NB_IMAGES * 0.8), NB_IMAGES))
            elif dataset_type == 'full':
                self._ids = list(range(NB_IMAGES))
        else:
            self._ids = ids
        if self.param('drop_end_sinos'):
            self._ids = list(filter(lambda x: x % self.DEPTH_PER_MICE <
                                    self.HIGH_NOISE_MAX_DEPTH, self._ids))
        self._keys = ['id', 'sinogram']

        dataset = self._create_dataset()
        dataset = self._processing_dataset(dataset)
        self._register_dataset(dataset)

    def _create_dataset(self):
        from functools import partial
        output_types = {k: data_type_tf(k) for k in self._keys}
        output_shapes = {k: data_shape(k) for k in self._keys}
        dataset_gen = partial(dataset_generator, ids=self._ids)
        return tf.data.Dataset.from_generator(dataset_gen, output_types, output_shapes)

    def _format_tensors(self, data):
        return {k: tf.reshape(tf.cast(data[k], data_type_tf(k)), data_shape(k)) for k in data}

    def _processing_dataset(self, dataset):
        dataset = (dataset
                   .repeat()
                   .cache()
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
