import tensorflow as tf
from dxpy.configs import configurable
from dxpy.learn.graph import Graph, NodeKeys
from dxpy.learn.config import config
from dxpy.learn.utils.tensor import shape_as_list
import numpy as np
from typing import List


class SuperResolutionExternalDataset(Graph):
    @configurable(config, with_name=True)
    def __init__(self,
                 name='dataset/external',
                 nb_down_sample: int=3,
                 input_shape: List[int]=None,
                 with_random_crop: bool=True,
                 with_white_normalization: bool=True,
                 with_poission_noise: bool=False,
                 with_noise_label: bool=False,
                 target_shape: List[int]=None,
                 mean: float=0.0,
                 std: float=1.0,
                 **kw):
        super().__init__(name,
                         with_poission_noise=with_poission_noise,
                         target_shape=target_shape,
                         with_white_normalization=with_white_normalization,
                         with_noise_label=with_noise_label,
                         with_random_crop=with_random_crop,
                         nb_down_sample=nb_down_sample,
                         input_shape=input_shape,
                         mean=mean,
                         std=std,
                         **kw)
        result = self._get_sinogram_external()
        for k in result:
            self.register_node(k, result[k])

    def _nb_downs(self, start=0, offset=0, *, is_ratio=False):
        result = range(start, self.param('nb_down_sample') + offset)
        if is_ratio:
            result = [2**i for i in result]
        else:
            result = list(result)
        return result

    def _make_input_place_holder(self):
        with tf.name_scope('input'):
            dataset = tf.placeholder(tf.float32,
                                     self.param('input_shape'),
                                     name='external_input')
            return dataset

    def _target_shape(self, dataset):
        batch_size = shape_as_list(dataset[0])
        return [batch_size] + list(self.param('target_shape')) + [1]

    def _get_sinogram_external(self):
        from dxpy.learn.model.normalizer import FixWhite
        from ..super_resolution import SuperResolutionDataset
        with tf.name_scope('external_dataset'):
            dataset = self._make_input_place_holder()
            self.register_node('external_place_holder', dataset)
            if self.param('with_poission_noise'):
                with tf.name_scope('add_with_poission_noise'):
                    noise = tf.random_poisson(dataset, shape=[])
                    dataset = tf.concat([noise, dataset], axis=0)
            if self.param('with_white_normalization'):
                dataset = FixWhite(name=self.name / 'fix_white',
                                   inputs=dataset,
                                   mean=self.param('mean'),
                                   std=self.param('std')).as_tensor()
            if self.param('with_random_crop'):
                dataset = tf.random_crop(dataset, self._target_shape(dataset))
            dataset = SuperResolutionDataset(self.name / 'super_resolution',
                                             lambda: {'image': dataset},
                                             input_key='image',
                                             nb_down_sample=self.param('nb_down_sample'))
            keys = ['image{}x'.format(2**i)
                    for i in range(dataset.param('nb_down_sample') + 1)]
            result = dict()
            if self.param('with_poission_noise'):
                result.update(
                    {'noise/' + k: dataset[k][:shape_as_list(dataset[k])[0] // 2, ...] for k in keys})
                result.update(
                    {'clean/' + k: dataset[k][shape_as_list(dataset[k])[0] // 2:, ...] for k in keys})
            else:
                result.update({'clean/' + k: dataset[k] for k in keys})
                result.update({'noise/' + k: dataset[k] for k in keys})
        return result
