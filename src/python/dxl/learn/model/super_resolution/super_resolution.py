import tensorflow as tf
import numpy as np
from typing import Dict
from dxl.learn.core import Model, Tensor
from dxl.learn.model.stack import Stack
from dxl.learn.model.residual import Residual
from dxl.learn.model.crop import boundary_crop, align_crop, shape_as_list
from dxl.learn.model.cnn import UpSampling2D, Conv2D
from dxl.learn.model.losses import mean_square_error, CombinedSupervisedLoss, poission_loss

__all__ = [
    "SuperResolution2x",
    "SuperResolutionBlock"
]


class SRKeys:
    REPRESENTS = 'reps'
    RESIDUAL = 'resi'
    ALIGNED_LABEL = 'aligned_label'
    INTERP = 'interp'
    POI_LOSS = 'poi_loss'
    MSE_LOSS = 'mse_loss'


class SuperResolution2x(Model):
    """ SuperResolution2x Block
    Arguments:
        name: Path := dxl.fs
            A unique block name
        inputs: Dict[str, Tensor/tf.Tensor] input.
        nb_layers: integer.
        filters: Integer, the dimensionality of the output space.
        boundary_crop: Tuple/List of 2 integers.
        graph: kernel
            One of StackedConv2D/StackedResidualConv/StackedResidualIncept
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            INFERENCE = 'inference'
            LABEL = 'label'
            LOSS = 'loss'

        class CONFIG:
            NB_LAYERS = 'nb_layers'
            FILTERS = 'filters'
            BOUNDARY_CROP = 'boundary_crop'

        class GRAPHS:
            SHORT_CUT = 'buildingblock'

    def __init__(self,
                 info,
                 inputs,
                 nb_layers=None,
                 filters=None,
                 boundary_crop=None,
                 graph=None):
        super().__init__(
            info,
            tensors=inputs,
            graphs={self.KEYS.GRAPHS.SHORT_CUT: graph},
            config={
                self.KEYS.CONFIG.NB_LAYERS: nb_layers,
                self.KEYS.CONFIG.FILTERS: filters,
                self.KEYS.CONFIG.BOUNDARY_CROP: boundary_crop,
            })

    @classmethod
    def _default_config(cls):
        return {
            cls.KEYS.CONFIG.NB_LAYERS: 2,
            cls.KEYS.CONFIG.FILTERS: 5,
            cls.KEYS.CONFIG.BOUNDARY_CROP: (4, 4)
        }

    def _short_cut(self, name):
        conv2d_ins = Conv2D(
            info="conv2d",
            filters=self.config(self.KEYS.CONFIG.FILTERS),
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='basic')
        return Stack(
            info=self.info.child_scope(name),
            models=conv2d_ins,
            nb_layers=2)

    def kernel(self, inputs):
        with tf.variable_scope('input'):
            u = UpSampling2D(
                inputs=inputs[self.KEYS.TENSOR.INPUT], size=(2, 2))()
            if SRKeys.REPRESENTS in inputs:
                r = UpSampling2D(
                    inputs=inputs[SRKeys.REPRESENTS], size=(2, 2))()
                r = align_crop(r, u)
                r = tf.concat([r, u], axis=3)
            else:
                r = tf.layers.conv2d(
                    inputs=u,
                    filters=self.config(self.KEYS.CONFIG.FILTERS),
                    kernel_size=5,
                    name='stem0')

        key = self.KEYS.GRAPHS.SHORT_CUT
        x = self.get_or_create_graph(key, self._short_cut(key))(r)
        with tf.variable_scope('inference'):
            res = tf.layers.conv2d(
                inputs=x,
                filters=1,
                kernel_size=3,
                padding='same',
                name='stem1',
            )
            res = boundary_crop(res,
                                self.config(self.KEYS.CONFIG.BOUNDARY_CROP))
            u_c = align_crop(u, res)
            y = res + u_c

        result = {
            self.KEYS.TENSOR.INFERENCE: y,
            SRKeys.REPRESENTS: x,
            SRKeys.RESIDUAL: res,
            SRKeys.INTERP: u_c
        }
        if self.KEYS.TENSOR.LABEL in inputs:
            with tf.name_scope('loss'):
                aligned_label = align_crop(inputs[self.KEYS.TENSOR.LABEL], y)
                l = mean_square_error(aligned_label, y)
            result.update({
                self.KEYS.TENSOR.LOSS: l,
                SRKeys.ALIGNED_LABEL: aligned_label
            })

        return result


class SuperResolutionBlock(Model):
    '''SuperResolutionBlock
    Arguments:
        name: Path := dxl.fs
            A unique block name
        inputs: Dict[str, Tensor/tf.Tensor] input.
            KEYS.TENSOR.INPUT: input low resolution image.
            KEYS.TENSOR.LABEL: [Optional] high resolution label image.
            SRKeys.REPRESENTS: [Optional] low resolution feature representations.
        interp: bool.
            if True, return upsampling result directly.
        graph: kernel
            One of StackedConv2D/StackedResidualConv/StackedResidualIncept
    '''

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            INFERENCE = 'inference'
            LABEL = 'label'
            LOSS = 'loss'

        class CONFIG:
            FILTERS = 'filters'
            BOUNDARY_CROP = 'boundary_crop'
            UPSAMPLE_RATIO = 'upsample_ratio'
            BOUNDARY_CROP_RATIO = 'boundary_crop_ratio'
            WITH_POI_LOSS = 'with_poi_loss'
            POI_LOSS_WEIGHT = 'poi_loss_weight'
            MES_LOSS_WEIGHT = 'mse_loss_weight'
            USE_COMBINED_LOSS = 'use_combined_loss'
            DENORM_STD = 'denorm_std'
            DENORM_MEAN = 'denorm_mean'
            INTERP = 'interp'

        class GRAPHS:
            SHORT_CUT = 'buildingblock'

    def __init__(self,
                 info,
                 inputs,
                 interp=None,
                 graph=None,
                 filters: int = None,
                 boundary_crop=None,
                 upsample_ratio=None,
                 boundary_crop_ratio=None,
                 with_poi_loss=None,
                 poi_loss_weight=None,
                 mse_loss_weight=None,
                 use_combined_loss=None,
                 denorm_std=None,
                 denorm_mean=None):
        super().__init__(
            info,
            tensors=inputs,
            graphs={self.KEYS.GRAPHS.SHORT_CUT: graph},
            config={
                self.KEYS.CONFIG.INTERP: interp,
                self.KEYS.CONFIG.FILTERS: filters,
                self.KEYS.CONFIG.BOUNDARY_CROP: boundary_crop,
                self.KEYS.CONFIG.UPSAMPLE_RATIO: upsample_ratio,
                self.KEYS.CONFIG.BOUNDARY_CROP_RATIO: boundary_crop_ratio,
                self.KEYS.CONFIG.WITH_POI_LOSS: with_poi_loss,
                self.KEYS.CONFIG.POI_LOSS_WEIGHT: poi_loss_weight,
                self.KEYS.CONFIG.MES_LOSS_WEIGHT: mse_loss_weight,
                self.KEYS.CONFIG.USE_COMBINED_LOSS: use_combined_loss,
                self.KEYS.CONFIG.DENORM_STD: denorm_std,
                self.KEYS.CONFIG.DENORM_MEAN: denorm_mean
            })

    @classmethod
    def _default_config(cls):
        return {
            cls.KEYS.CONFIG.INTERP: False,
            cls.KEYS.CONFIG.FILTERS: 32,
            cls.KEYS.CONFIG.BOUNDARY_CROP: (4, 4),
            cls.KEYS.CONFIG.UPSAMPLE_RATIO: (2, 2),
            cls.KEYS.CONFIG.BOUNDARY_CROP_RATIO: 0.1,
            cls.KEYS.CONFIG.WITH_POI_LOSS: False,
            cls.KEYS.CONFIG.POI_LOSS_WEIGHT: 0.0,
            cls.KEYS.CONFIG.MES_LOSS_WEIGHT: 1.0,
            cls.KEYS.CONFIG.USE_COMBINED_LOSS: False,
            cls.KEYS.CONFIG.DENORM_STD: 1.0,
            cls.KEYS.CONFIG.DENORM_MEAN: 0.0
        }

    def _short_cut(self, name):
        conv2d_ins = Conv2D(
            info="conv2d",
            filters=self.config(self.KEYS.CONFIG.FILTERS),
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            activation='basic')
        return Stack(
            info=self.info.child_scope(name),
            models=conv2d_ins,
            nb_layers=2)

    def _input(self, inputs):
        with tf.variable_scope('input'):
            u = UpSampling2D(
                'upsampling_u',
                inputs=inputs[self.KEYS.TENSOR.INPUT],
                size=self.config(self.KEYS.CONFIG.UPSAMPLE_RATIO))()

            if self.KEYS.TENSOR.LABEL in inputs:
                l = inputs[self.KEYS.TENSOR.LABEL]
            else:
                l = None

            if self.config(self.KEYS.CONFIG.INTERP):
                return u, None, l

            if SRKeys.REPRESENTS in inputs:
                r = UpSampling2D(
                    'upsampling_r',
                    inputs=inputs[SRKeys.REPRESENTS],
                    size=self.config(self.KEYS.CONFIG.UPSAMPLE_RATIO))()
                r = align_crop(r, u)
            else:
                r = tf.layers.conv2d(
                    inputs=u,
                    filters=self.config(self.KEYS.CONFIG.FILTERS),
                    kernel_size=5,
                    padding='same',
                    name='stem',
                    reuse=tf.AUTO_REUSE)

            return u, r, l

    def _inference(self, represents, upsampled):
        with tf.variable_scope('inference'):
            upsampled = boundary_crop(
                input_=upsampled,
                offset=self.config(self.KEYS.CONFIG.BOUNDARY_CROP))
            if self.config(self.KEYS.CONFIG.INTERP):
                return {self.KEYS.TENSOR.INFERENCE: upsampled}
            residual = tf.layers.conv2d(
                inputs=represents,
                filters=1,
                kernel_size=3,
                padding='same',
                reuse=tf.AUTO_REUSE)
            residual = align_crop(input_=residual, target=upsampled)
            inference = residual + upsampled

        return {
            self.KEYS.TENSOR.INFERENCE: inference,
            SRKeys.RESIDUAL: residual,
            SRKeys.INTERP: upsampled
        }

    def _loss(self, label, infer):
        if label is not None:
            with tf.name_scope('loss'):
                align_label = align_crop(label, infer)
                if self.config(self.KEYS.CONFIG.USE_COMBINED_LOSS):
                    with tf.name_scope('use_combine_loss'):
                        stdv = tf.constant(
                            self.config(self.KEYS.CONFIG.DENORM_STD),
                            tf.float32)
                        meanv = tf.constant(
                            self.config(self.KEYS.CONFIG.DENORM_MEAN,
                                        tf.float32))
                        labeld = align_label * stdv + meanv
                        inferd = infer * stdv + meanv
                    result = CombinedSupervisedLoss(
                        self.name / 'loss',
                        inputs={
                            self.KEYS.TENSOR.INPUT: inferd,
                            self.KEYS.TENSOR.LABEL: labeld
                        })()
                    result.update({SRKeys.ALIGNED_LABEL: align_label})
                    result.update({
                        self.KEYS.TENSOR.LOSS:
                        result[self.KEYS.TENSOR.OUTPUT]
                    })
                    result.pop(self.KEYS.TENSOR.OUTPUT)
                    return result
                else:
                    loss_mse = mean_square_error(
                        align_label, infer) * self.config(
                            self.KEYS.CONFIG.MES_LOSS_WEIGHT)
                    loss = loss_mse
                    if self.config(self.KEYS.CONFIG.WITH_POI_LOSS):
                        loss_poi = poission_loss(
                            align_label, infer) * self.config(
                                self.KEYS.CONFIG.POI_LOSS_WEIGHT)
                        loss = loss + loss_poi
                    result = {
                        self.KEYS.TENSOR.LOSS: loss,
                        SRKeys.MSE_LOSS: loss_mse,
                        SRKeys.ALIGNED_LABEL: align_label
                    }
                    return result
        else:
            return {}

    def kernel(self, inputs):
        upsampled, represents, label = self._input(inputs)
        if self.config(self.KEYS.CONFIG.INTERP):
            return upsampled

        key = self.KEYS.GRAPHS.SHORT_CUT
        x = self.get_or_create_graph(key, self._short_cut(key))(represents)
        result = {SRKeys.REPRESENTS: x}
        result.update(self._inference(x, upsampled))
        result.update(self._loss(label, result[self.KEYS.TENSOR.INFERENCE]))

        return result
