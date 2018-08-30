# -*- coding: utf-8 -*-

import tensorflow as tf

from .base import Model
from doufo import List
from doufo.collections import concatenate

__all__ = [
    # 'Conv1D',
    'Conv2D',
    'Inception',
    # 'Conv3D',
    # 'DeConv2D',
    # 'DeConv3D',
    # 'UpSampling2D',
    # 'DownSampling2D',
    # 'DeformableConv2D',
    # 'AtrousConv1D',
    # 'AtrousConv2D',
    # 'deconv2D_bilinear_upsampling_initializer',
    # 'DepthwiseConv2D',
    # 'SeparableConv2D',
    # 'GroupConv2D',
]


class Conv2D(Model):
    """2D convolution model
    Arguments:
        name: Path := dxl.fs.
        inputs: Tensor input.
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers.
        strides: An integer or tuple/list of 2 integers.
        padding: One of "valid" or "same" (case-insensitive).
        activation: Activation function. Set it to None to maintain a linear activation.
        graph_info: GraphInfo or DistributeGraphInfo
    """
    _nargs = 1
    _nouts = 1

    class KEYS(Model.KEYS):
        class CONFIG:
            FILTERS = 'filters'
            KERNEL_SIZE = 'kernel_size'
            STRIDES = 'strides'
            PADDING = 'padding'

    def __init__(self, name,
                 filters=None, kernel_size=None, strides=None, padding=None):
        super().__init__(name)
        self._config[self.KEYS.CONFIG.FILTERS] = filters
        self._config[self.KEYS.CONFIG.KERNEL_SIZE] = kernel_size
        self._config[self.KEYS.CONFIG.STRIDES] = strides if strides is not None else (1, 1)
        self._config[self.KEYS.CONFIG.PADDING] = padding if padding is not None else 'same'
        self.model = None

    def build(self, x):
        if isinstance(x, tf.Tensor):
            self.model = tf.layers.Conv2D(self.config[self.KEYS.CONFIG.FILTERS],
                                          self.config[self.KEYS.CONFIG.KERNEL_SIZE],
                                          self.config[self.KEYS.CONFIG.STRIDES],
                                          self.config[self.KEYS.CONFIG.PADDING])
        raise TypeError(f"Not support tensor type: {type(x)}.")

    def kernel(self, x):
        return self.model(x)

    @property
    def parameters(self):
        return self.model.weights


class Inception(Model):
    """InceptionBlock model
    Arguments:
        name: Path := dxl.fs.
        paths: List[Model].
        graph_info: GraphInfo or DistributeGraphInfo
    """
    _nargs = 1
    _nouts = 1

    def __init__(self, name, init_op: Model, paths, merge):
        super().__init__(name)
        self.init_op = init_op
        self.paths = paths
        self.merge = merge

    def kernel(self, x):
        x = self.init_op(x)
        return self.merge([p(x) for p in self.paths])

    @property
    def parameters(self):
        models = List()
        if isinstance(self.init_op, Model):
            models.append(self.init_op)
        models += List(self.paths).filter(lambda m: isinstance(m, Model))
        if isinstance(self.merge, Model):
            models.append(self.merge)
        return concatenate(models.fmap(lambda m: m.parameters))
