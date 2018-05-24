import os
import tensorflow as tf
from tables import *
from dxpy.core.path import Path
from dxpy.configs import configurable
import h5py
import numpy as np
DEFAULT_FILE_NAME = 'analytical_phantom_sinogram.h5'
from ...config import get_config
from ..base import DatasetFromTFDataset, DatasetFromGenerator


class AnalyticalPhantomSinogram(IsDescription):
    phantom = UInt16Col(shape=(256, 256))
    sinogram = UInt16Col(shape=(320, 320))
    phantom_type = UInt8Col()


class AnalyticalPhantomMultiScaleReconstruction(IsDescription):
    recon1x = UInt16Col(shape=(256, 256))
    recon2x = UInt16Col(shape=(256, 256))
    recon4x = UInt16Col(shape=(256, 256))
    recon8x = UInt16Col(shape=(256, 256))


def data_type(field_name):
    import tensorflow as tf
    return {
        'phantom': np.float32,
        'sinogram': np.float32,
        'phantom_type': np.int64
    }[field_name]


def data_type_tf(field_name):
    import tensorflow as tf
    return {
        'phantom': tf.float32,
        'sinogram': tf.float32,
        'phantom_type': tf.int64
    }[field_name]


def data_shape(field_name):
    import tensorflow as tf
    return {
        'phantom': (256, 256),
        'sinogram': (640, 320),
        'phantom_type': (),
    }[field_name]


def h5filename():
    from ..config import config
    return str(Path(config['PATH_DATASETS']) / DEFAULT_FILE_NAME)


def _post_processing(result):
    from dxpy.medical_image_processing.projection.parallel import padding_pi2full
    if 'sinogram' in result:
        result['sinogram'] = padding_pi2full(result['sinogram']).T
    for k in result:
        result[k] = result[k].astype(data_type(k))
    return result


def dataset(fields=('phantom', 'sinogram'), filename=None, idxs=None):
    with open_file(h5filename()) as h5file:
        if idxs is None:
            rows = h5file.root.data.iterrows()
        else:
            rows = idxs
        for d in rows:
            if isinstance(d, int):
                d = h5file.root.data[d]
            result = {k: d[k] for k in fields}
            result = _post_processing(result)
            yield result


def dataset_multi_image(fields=('phantom', 'sinogram', 'recon1x', 'recon2x', 'recon4x', 'recon8x'), filename_sinogram=None, filename_recon=None, idxs=None):
    if filename_recon is None:
        filename_recon = ''
    # TODO: finish
    # 施工中


def _tensorflow_raw_dataset(fields, idxs=None):
    import tensorflow as tf
    from functools import partial
    dataset_gen = partial(dataset, fields=fields, idxs=idxs)
    return tf.data.Dataset.from_generator(dataset_gen, {k: data_type_tf(k) for k in fields}, {k: data_shape(k) for k in fields})


class AnalyticalPhantomSinogramDataset(DatasetFromGenerator):
    @configurable(get_config(), with_name=True)
    def __init__(self, name='dataset', batch_size=32, fields=('sinogram', 'phantom'), idxs=None, shuffle=True):
        from functools import partial
        dataset_gen = partial(dataset, fields=fields, idxs=idxs)
        output_types = {k: data_type_tf(k) for k in fields}
        output_shapes = {k: data_shape(k) for k in fields}
        super().__init__(name, dataset_gen, batch_size=batch_size, idxs=idxs,
                         output_types=output_types, output_shapes=output_shapes, shuffle=shuffle)

    def _reshape_tensors(self, data):
        return {'sinogram': tf.reshape(tf.cast(data['sinogram'], tf.float32),
                                       data_shape('sinogram')),
                'phantom': tf.reshape(tf.cast(data['phantom'], tf.float32),
                                      data_shape('phantom'))}

    def _processing(self):
        dataset = (super()._processing()
                   .repeat()
                   .map(self._reshape_tensors))
        if self.param('shuffle'):
            dataset = dataset.shuffle(1024)
        return dataset.batch(self.param('batch_size'))


@configurable(get_config(), with_name=True)
def get_dataset(name='dataset', fields=['sinogram']):
    from ..base import DatasetFromTFDataset
    return DatasetFromTFDataset(name, _tensorflow_raw_dataset(fields))

import typing
@configurable(get_config(), with_name=True)
def analytical_super_resolution_dataset(image_type: str,
                                        poission_noise: bool,
                                        batch_size: int,
                                        nb_down_sample: int,
                                        target_shape: typing.List[int],
                                        idxs: typing.List[int],
                                        *,
                                        name: str='dataset',
                                        path_dataset: "str|None"=None):
    """
    Args:
        -   image_type: 'sinogram' or 'image'
        -   batch_size
    Returns:
        a `Graph` object, which has several nodes:
    Raises:
    """
    from ...model.normalizer.normalizer import FixWhite, ReduceSum
    from ..super_resolution import SuperResolutionDataset
    from ...model.tensor import ShapeEnsurer
    from ...config import config
    normalizer_configs = {
        'analytical_phantoms': {'mean': 4.88, 'std': 4.68}
    }
    config_origin = {}
    config_normalizer = normalizer_configs[dataset_name]
    config['dataset'] = {
        'origin': config_origin,
        'fix_white': config_normalizer
    }
    with tf.name_scope('{img_type}_dataset'.format(img_type=image_type)):
        if image_type == 'sinogram':
            dataset_origin = AnalyticalPhantomSinogramDataset(name=name, batch_size=batch_size, fields=['sinogram'], idxs)
        dataset_summed = ReduceSum('dataset/reduce_sum',
                                   dataset_origin[image_type],
                                   fixed_summation_value=1e6).as_tensor()

        dataset = FixWhite(name='dataset/fix_white',
                           inputs=dataset_summed)()
        if poission_noise:
            with tf.name_scope('add_poission_noise'):
                dataset = tf.random_poisson(dataset, shape=[])
        dataset = tf.random_crop(dataset,
                                 [batch_size] + list(target_shape) + [1])
        dataset = SuperResolutionDataset('dataset/super_resolution',
                                         lambda: {'image': dataset},
                                         input_key='image',
                                         nb_down_sample=3)
    return dataset




# The following part is a record of dataset generator script.


# def create_dataset_from_tf_records():
#     from tqdm import tqdm
#     from dxpy.learn.dataset.config import config
#     filename = '/media/hongxwing/73f574f4-33b0-46b1-ae5d-de1eca4ef891/home/hongxwing/Workspace/Datas/phantom_sinograms_3.h5'
#     with h5py.File(filename, 'r') as fin:
#         filters = Filters(5, 'blosc')
#         h5file = open_file(str(Path(config['PATH_DATASETS']) / DEFAULT_FILE_NAME),
#                            mode="w", filters=filters)
#         table = h5file.create_table(
#             h5file.root, 'data', AnalyticalPhantomSinogram, expectedrows=fin['phantoms'].shape[0])
#         data = table.row

#         def convert_to_uint16(arr):
#             UINT16MAX = 65535
#             arr = np.maximum(arr, 0.0)
#             arr = arr / np.max(arr) * UINT16MAX
#             arr = np.minimum(arr, UINT16MAX)
#             return arr.astype(np.uint32)

#         for i in tqdm(range(fin['phantoms'].shape[0]), ascii=True):
#             data['sinogram'] = convert_to_uint16(
#                 fin['sinograms'][i, :, :320, 0])
#             data['phantom'] = convert_to_uint16(fin['phantoms'][i, :, :, 0])
#             data['phantom_type'] = np.reshape(np.array(fin['types'][i, ...],
#                                                        dtype=np.uint8), [])
#             data.append()
#     table.flush()
#     h5file.close()


# if __name__ == "__main__":
#     create_dataset_from_tf_records()
