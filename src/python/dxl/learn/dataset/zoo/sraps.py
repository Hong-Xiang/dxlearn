"""
Super resolution dataset for analytical phantom sinogram dataset
"""
from typing import List

import tensorflow as tf

from dxpy.configs import configurable

from ...config import config
from ...graph import Graph, NodeKeys
from dxpy.learn.utils.tensor import shape_as_list
import numpy as np


class AnalyticalPhantomSinogramDatasetForSuperResolution(Graph):
    @configurable(config, with_name=True)
    def __init__(self, name='datasets/apssr',
                 image_type: str='sinogram',
                 nb_down_sample: int=3,
                 *,
                 log_scale: bool=False,
                 with_white_normalization: bool=True,
                 with_poission_noise: bool=False,
                 with_noise_label: bool=False,
                 with_phase_shift: bool=False,
                 target_shape: List[int]=None,
                 low_dose: bool= False,
                 low_dose_ratio: float= 10.0,
                 **kw):
        """
        Args:
            -   image_type: 'sinogram' or 'image'
            -   log_scale: produce log datas
            -   with_white_normalization: normalize by mean and std
            -   with_poission_noise (apply to sinogram only): perform poission sample
            -   target_shape: (apply to sinogram only), random crop shape
        Returns:
            a `Graph` object, which has several nodes:
        Raises:
        """
        super().__init__(name, image_type=image_type,
                         log_scale=log_scale,
                         with_poission_noise=with_poission_noise,
                         target_shape=target_shape,
                         with_white_normalization=with_white_normalization,
                         with_noise_label=with_noise_label,
                         with_phase_shift=with_phase_shift,
                         low_dose=low_dose,
                         low_dose_ratio=low_dose_ratio,
                         nb_down_sample=nb_down_sample, **kw)
        # from ...model.normalizer.normalizer import FixWhite, ReduceSum
        # from ...model.tensor import ShapeEnsurer
        # from dxpy.core.path import Path
        # name = Path(name)
        # if image_type == 'sinogram':
        #     fields = ['sinogram', 'id']
        # elif image_type == 'sinogram_with_phantom':
        #     fields = ['sinogram', 'phantom', 'id']
        # else:
        #     fields = ['phantom', 'id'] + ['recon{}x'.format(2**i)
        #                                   for i in range(self.param('nb_down_sample') + 1)]
        # with tf.name_scope('aps_{img_type}_dataset'.format(img_type=image_type)):
        #     dataset_origin = Dataset(
        #         self.name / 'analytical_phantom_sinogram', fields=fields)
        #     tid = dataset_origin['id']
        #     if image_type in ['sinogram', 'sinogram_with_phantom']:
        #         dataset = self._process_sinogram(dataset_origin)
        #     else:
        #         dataset = self._process_recons(dataset_origin)
        # # for k in dataset
        #     # self.register_node(k, dataset[k]):

        # # Old fashion
        # for i in range(self.param('nb_down_sample') + 1):
        #     img_key = 'image{}x'.format(2**i)
        #     self.register_node('noise/' + img_key, dataset['noise/' + img_key])
        #     self.register_node('clean/' + img_key, dataset['clean/' + img_key])
        #     self.register_node('input/image{}x'.format(2**i),
        #                        dataset['input/image{}x'.format(2**i)])
        #     if self.param('image_type') == 'image' and i == 0:
        #         self.register_node('label/image1x', dataset['label/phantom'])
        #         continue
        #     if self.param('with_noise_label'):
        #         self.register_node('label/image{}x'.format(2**i),
        #                            dataset['input/image{}x'.format(2**i)])
        #     else:
        #         self.register_node('label/image{}x'.format(2**i),
        #                            dataset['label/image{}x'.format(2**i)])

        if image_type in ['sinogram', 'sinogram_with_phantom', 'mice_sinograms']:
            if image_type in ['sinogram', 'sinogram_with_phantom']:
                result = self._get_sinogram_aps()
            else:
                result = self._get_sinogram_mice()
            if self.param('with_poission_noise'):
                input_keys = [
                    'noise/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
            else:
                input_keys = [
                    'clean/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
            if self.param('with_noise_label'):
                label_keys = [
                    'noise/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
            else:
                label_keys = [
                    'clean/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
        elif image_type == 'recons_full_size':
            result = self._get_phantom_aps()
            input_keys = [
                'noise/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
            label_keys = [
                'noise/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
        elif image_type == 'mCT':
            result = self._get_sinogram_mct()
            input_keys = [
                'noise/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
            label_keys = [
                'noise/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
        elif image_type == 'recon_ms':
            result = self._get_recons_ms()
            if self.param('with_poission_noise'):
                input_keys = [
                    'noise{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
            else:
                input_keys = [
                    'clean{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
            if self.param('with_noise_label'):
                label_keys = [
                    'noise{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
            else:
                label_keys = [
                    'clean{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
        else:
            raise ValueError('Unknown image type {}.'.format(image_type))
        output_input_keys = [
            'input/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
        output_label_keys = [
            'label/image{}x'.format(2**i) for i in range(self.param('nb_down_sample') + 1)]
        for ki, ko in zip(input_keys, output_input_keys):
            self.register_node(ko, result[ki])
        for ki, ko in zip(label_keys, output_label_keys):
            self.register_node(ko, result[ki])

    def _get_sinogram_aps(self):
        from ..raw.analytical_phantom_sinogram import Dataset
        from dxpy.learn.model.normalizer import FixWhite
        from ..super_resolution import SuperResolutionDataset
        fields = ['sinogram', 'id', 'phantom']
        with tf.name_scope('aps_sinogram_dataset'):
            dataset = Dataset(
                self.name / 'analytical_phantom_sinogram', fields=fields)
            self.register_node('id', dataset['id'])
            self.register_node('phantom', dataset['phantom'])
            if self.param('log_scale'):
                stat = dataset.LOG_SINO_STAT
            else:
                stat = dataset.SINO_STAT
            dataset = dataset['sinogram']
            if self.param('with_poission_noise'):
                with tf.name_scope('add_with_poission_noise'):
                    noise = tf.random_poisson(dataset, shape=[])
                    dataset = tf.concat([noise, dataset], axis=0)
                if self.param('log_scale'):
                    dataset = tf.log(dataset + 0.4)
            if self.param('with_white_normalization'):
                dataset = FixWhite(name=self.name / 'fix_white',
                                   inputs=dataset, mean=stat['mean'], std=stat['std']).as_tensor()
            dataset = tf.random_crop(dataset,
                                     [shape_as_list(dataset)[0]] + list(self.param('target_shape')) + [1])
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


    def _get_recons_ms(self):
        from ..raw.analytical_phantom_sinogram import Dataset
        from dxpy.learn.model.normalizer import FixWhite
        ratios = [2**i for i in range(self.param('nb_down_sample') + 1)]
        fields = ['phantom', 'id'] + ['clean{}x'.format(r) for r in ratios] + [
            'noise{}x'.format(r) for r in ratios]
        with tf.name_scope('dataset_recon_ms'):
            dataset = Dataset(
                self.name / 'analytical_phantom_sinogram', fields=fields)
            self.register_node('id', dataset['id'])
            self.register_node('phantom', dataset['phantom'])
            keys_noise = ['noise{}x'.format(r) for r in ratios]
            keys_clean = ['clean{}x'.format(r) for r in ratios]
            stat = dataset.RECON_MS_STAT
            cleans = dict()
            noises = dict()
            if self.param('with_white_normalization'):
                with tf.name_scope('normalizatoin_clean'):
                    for i, k in enumerate(keys_clean):
                        cleans[k] = FixWhite(name=self.name / 'fix_white', inputs=dataset[k],
                                             mean=stat['mean'] * (4.0)**i, std=stat['std'] * (4.0)**i).as_tensor()
                with tf.name_scope('normalizatoin_noise'):
                    for i, k in enumerate(keys_noise):
                        noises[k] = FixWhite(name=self.name / 'fix_white', inputs=dataset[k],
                                             mean=stat['mean'] * (4.0)**i, std=stat['std'] * (4.0)**i).as_tensor()
            # for i, r in ratios:
            #     self.register_node('clean/image{}x'.format(r), cleans[keys_clean[i]])
            #     self.register_node('noise/image{}x'.format(r), noises[keys_noise[i]])
            # if self.param('with_poission_noise'):
            #     images = {'input/image{}x'.format(r): noises['noise{}x'.format(r)] for r in ratios}
            # else:
            #     images = {'input/image{}x'.format(r): cleans['clean{}x'.format(r)] for r in ratios}
            # if self.param('with_noise_lable'):
            #     labels = {'label/image{}x'.format(r): noises['noise{}x'.format(r)] for r in ratios}
            # else:
            #     labels = {'label/image{}x'.format(r): cleans['clean{}x'.format(r)] for r in ratios}
            result = dict()
            result.update(cleans)
            result.update(noises)
            return result

    def _get_phantom_aps(self):
        from ..raw.analytical_phantom_sinogram import Dataset
        from ..super_resolution import SuperResolutionDataset
        fields = ['phantom', 'id'] + ['recon{}x'.format(2**i)
                                      for i in range(self.param('nb_down_sample') + 1)]
        with tf.name_scope('aps_phantom_dataset'):
            dataset_origin = Dataset(
                self.name / 'analytical_phantom_sinogram', fields=fields)
            self.register_node('id', dataset_origin['id'])
            self.register_node('phantom', dataset['phantom'])
            label = dataset['phantom']
            keys = ['recon{}x'.format(2**i)
                    for i in range(self.param('nb_down_sample') + 1)]
            dataset = dataset_origin
            if self.param('log_scale'):
                stat = dataset.LOG_RECON_STAT
            else:
                stat = dataset.RECON_STAT
            dataset = {k: dataset[k] for k in keys}
            for i, k in enumerate(keys):
                if i == 0:
                    continue
                dataset[k] = tf.nn.avg_pool(dataset[k],
                                            [1] + [2**i, 2**i] + [1],
                                            padding='SAME',
                                            strides=[1] + [2**i, 2**i] + [1]) * (2.0**i * 2.0**i)
            if self.param('log_scale'):
                for k in keys:
                    dataset[k] = tf.log(dataset[k] + 1.0)
                label = tf.log(label + 1.0)
            if self.param('with_white_normalization'):
                for i, k in enumerate(keys):
                    dataset[k] = FixWhite(name=self.name / 'fix_white' / k,
                                          inputs=dataset[k], mean=stat['mean'], std=stat['std']).as_tensor()
                label = FixWhite(name=self.name / 'fix_white' / 'phantom',
                                 inputs=dataset[k], mean=stat['mean'], std=stat['std']).as_tensor()
            result = dict()
            for i, k in enumerate(keys):
                result['noise/image{}x'.format(2**i)] = dataset[k]
            result['label/phantom'] = phantom
            return result

    def _get_sinogram_mct(self):
        from ..raw.mCT import Dataset
        from dxpy.learn.model.normalizer import FixWhite
        from ..super_resolution import random_crop_multiresolution
        with tf.variable_scope('mCT_dataset'):
            dataset_origin = Dataset(self.name / 'mCT')
            self.register_node('id', dataset_origin['id'])
            self.register_node('phantom', dataset_origin['phantom'])
            dataset = dict()
            keys = ['noise/image{}x'.format(2**i)
                    for i in range(self.param('nb_down_sample') + 1)]
            for i, k in enumerate(keys):
                with tf.variable_scope('normalization_{}x'.format(2**i)):
                    fwn = FixWhite(self.name / 'normalization_{}x'.format(2**i),
                                   inputs=dataset_origin['sinogram{}x'.format(
                                       2**i)],
                                   mean=dataset_origin.MEAN * (4.0**i),
                                   std=dataset_origin.STD * (4.0**i))
                    dataset['noise/image{}x'.format(2**i)] = fwn.as_tensor()
                    # dataset[k] = dataset_origin['sinogram{}x'.format(2**i)]
            crop_keys = ['image{}x'.format(i) for i in range(len(keys))]
            images = {'image{}x'.format(
                i): dataset[k] for i, k in enumerate(keys)}
            images_cropped = random_crop_multiresolution(images,
                                                         [shape_as_list(images['image1x'])[0]] + list(self.param('target_shape')) + [1])

            result = dict()
            for ki, ko in zip(crop_keys, keys):
                result[ko] = images_cropped[ki]
            return result

    def _get_sinogram_mice(self):
        from ..raw.mice import Dataset
        from dxpy.learn.model.normalizer import FixWhite
        from ..super_resolution import SuperResolutionDataset
        with tf.name_scope('mice_sinogram_dataset'):
            dataset = Dataset(self.name / 'mice_sinogram')
            self.register_node('id', dataset['id'])
            bs = dataset.param('batch_size')
            stat = {'mean': dataset.MEAN, 'std': dataset.STD}
            dataset = dataset['sinogram']
            if self.param('with_poission_noise'):
                with tf.name_scope('add_with_poission_noise'):
                    if self.param('low_dose'):
                        ratio = self.param('low_dose_ratio')
                        ratio_norm = 4e6 * bs / ratio
                        dataset = dataset / tf.reduce_sum(dataset) * ratio_norm
                        stat['mean'] = stat['mean'] / ratio
                        stat['std'] = stat['std'] / ratio
                    else:
                        dataset = dataset / tf.reduce_sum(dataset) * 4e6 * bs
                    noise = tf.random_poisson(dataset, shape=[])
                    dataset = tf.concat([noise, dataset], axis=0)
            if self.param('with_white_normalization'):
                dataset = FixWhite(name=self.name / 'fix_white',
                                   inputs=dataset, mean=stat['mean'], std=stat['std']).as_tensor()
            dataset = tf.random_crop(dataset,
                                     [shape_as_list(dataset)[0]] + list(self.param('target_shape')) + [1])
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

    def _process_sinogram(self, dataset):
        from ...model.normalizer.normalizer import ReduceSum, FixWhite
        from dxpy.learn.model.normalizer import FixWhite
        from ..super_resolution import SuperResolutionDataset
        from ...utils.tensor import shape_as_list
        if self.param('log_scale'):
            stat = dataset.LOG_SINO_STAT
        else:
            stat = dataset.SINO_STAT
        # dataset = ReduceSum(self.name / 'reduce_sum', dataset['sinogram'],
            # fixed_summation_value=1e6).as_tensor()
        if 'phantom' in dataset:
            phan = dataset['phantom']
        else:
            phan = None
        dataset = dataset['sinogram']
        if self.param('with_poission_noise'):
            with tf.name_scope('add_with_poission_noise'):
                noise = tf.random_poisson(dataset, shape=[])
                dataset = tf.concat([noise, dataset], axis=0)
        if self.param('log_scale'):
            dataset = tf.log(dataset + 0.4)
        if self.param('with_white_normalization'):
            dataset = FixWhite(name=self.name / 'fix_white',
                               inputs=dataset, mean=stat['mean'], std=stat['std']).as_tensor()
        # random phase shift
        # if self.param('with_phase_shift'):
        #     phase_view = tf.random_uniform(
        #         [], 0, shape_as_list(dataset)[1], dtype=tf.int64)
        #     dataset_l = dataset[:, phase_view:, :, :]
        #     dataset_r = dataset[:, :phase_view, :, :]
        #     dataset = tf.concat([dataset_l, dataset_r], axis=1)
        dataset = tf.random_crop(dataset,
                                 [shape_as_list(dataset)[0]] + list(self.param('target_shape')) + [1])
        dataset = SuperResolutionDataset(self.name / 'super_resolution',
                                         lambda: {'image': dataset},
                                         input_key='image',
                                         nb_down_sample=self.param('nb_down_sample'))
        keys = ['image{}x'.format(2**i)
                for i in range(dataset.param('nb_down_sample') + 1)]
        if self.param('with_poission_noise'):
            result = {
                'input/' + k: dataset[k][:shape_as_list(dataset[k])[0] // 2, ...] for k in keys}
            result.update(
                {'label/' + k: dataset[k][shape_as_list(dataset[k])[0] // 2:, ...] for k in keys})
        else:
            result = {'input/' + k: dataset[k] for k in keys}
            result.update({'label/' + k: dataset[k] for k in keys})
        result.update(
            {'noise/' + k: dataset[k][:shape_as_list(dataset[k])[0] // 2, ...] for k in keys})
        result.update(
            {'clean/' + k: dataset[k][shape_as_list(dataset[k])[0] // 2:, ...] for k in keys})
        if phan is not None:
            result.update({'phantom': phan})
        return result

    def _process_recons(self, dataset):
        from ...model.normalizer.normalizer import ReduceSum, FixWhite
        from dxpy.learn.model.normalizer import FixWhite
        from ..super_resolution import SuperResolutionDataset
        from ...utils.tensor import shape_as_list
        keys = ['recon{}x'.format(2**i)
                for i in range(self.param('nb_down_sample') + 1)]
        if self.param('log_scale'):
            stat = dataset.LOG_RECON_STAT
        else:
            stat = dataset.RECON_STAT
        phantom = dataset['phantom']
        dataset = {k: dataset[k] for k in keys}
        for i, k in enumerate(keys):
            if i == 0:
                continue
            dataset[k] = tf.nn.avg_pool(dataset[k],
                                        [1] + [2**i, 2**i] + [1],
                                        padding='SAME',
                                        strides=[1] + [2**i, 2**i] + [1]) * (2.0**i * 2.0**i)

        # dataset = {k: ReduceSum(self.name / 'reduce_sum' / k, dataset[k], fixed_summation_value=1e6).as_tensor() for k in keys}
        if self.param('log_scale'):
            for k in keys:
                dataset[k] = tf.log(dataset[k] + 1.0)
            phantom = tf.log(phantom + 1.0)
        if self.param('with_white_normalization'):
            for i, k in enumerate(keys):
                dataset[k] = FixWhite(name=self.name / 'fix_white' / k,
                                      inputs=dataset[k], mean=stat['mean'], std=stat['std']).as_tensor()
            phantom = FixWhite(name=self.name / 'fix_white' / 'phantom',
                               inputs=dataset[k], mean=stat['mean'], std=stat['std']).as_tensor()

        result = dict()
        for i, k in enumerate(keys):
            result['input/image{}x'.format(2**i)] = dataset[k]
            result['label/image{}x'.format(2**i)
                   ] = result['input/image{}x'.format(2**i)]
        result['label/phantom'] = phantom
        return result


