# -*- coding: utf-8 -*-

import tensorflow as tf

from dxl.learn.model import Model
from dxl.learn.model.merge import Merge
from doufo import List
from doufo.collections.concatenate import concat
from functools import partial

__all__ = [
    # 'Conv1D',
    'Conv2D',
    'Inception',
    # 'Conv3D',
    # 'DeConv2D',
    # 'DeConv3D',
    'UpSampling2D',
    'DownSampling2D',
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

    def __init__(self, name, filters=None, kernel_size=None, strides=None, padding=None,
                 ):
        super().__init__(name)
        self.config.update(self.KEYS.CONFIG.FILTERS, filters)
        self.config.update(self.KEYS.CONFIG.KERNEL_SIZE, kernel_size)
        self.config.update_value_and_default(self.KEYS.CONFIG.STRIDES, strides, (1, 1))
        self.config.update_value_and_default(self.KEYS.CONFIG.PADDING, padding, 'same')
        self.model = None

    def build(self, x):
        if isinstance(x, tf.Tensor):
            self.model = tf.layers.Conv2D(self.config[self.KEYS.CONFIG.FILTERS],
                                          self.config[self.KEYS.CONFIG.KERNEL_SIZE],
                                          self.config[self.KEYS.CONFIG.STRIDES],
                                          self.config[self.KEYS.CONFIG.PADDING])
            return
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

    def __init__(self, name, init_op: Model, merge, paths):
        super().__init__(name)
        self.init_op = init_op
        self.paths = paths
        self.merge = merge
        self.model = None

    def kernel(self, x):
        x = self.init_op(x)
        return self.merge(self.model(x))

    def build(self, x):
        if isinstance(x, tf.Tensor):
            new_merger = partial(concat, axis=3)
            self.model = Merge(merger=new_merger, models=self.paths)
            return
        raise NotImplementedError(f"Incept not implemented for {type(x)}.")

    @property
    def parameters(self):
        models = List()
        if isinstance(self.init_op, Model):
            models.append(self.init_op)
        models += List(self.paths).filter(lambda m: isinstance(m, Model))
        if isinstance(self.merge, Model):
            models.append(self.merge)
        return concat(models.fmap(lambda m: m.parameters))


class DownSampling2D(Model):
    class KEYS(Model.KEYS):
        class CONFIG:
            POOL_SIZE = 'pool_size'
            STRIDE = 'stride'
            PADDING = 'padding'
            METHOD = 'method'

    def __init__(self,
                 name,
                 pool_size=None,
                 stride=None,
                 padding=None,
                 method=None):
        super().__init__(name)
        self.config.update_value_and_default(self.KEYS.CONFIG.POOL_SIZE, pool_size, (1, 1))
        self.config.update_value_and_default(self.KEYS.CONFIG.STRIDE, stride, (1, 1))
        self.config.update_value_and_default(self.KEYS.CONFIG.PADDING, padding, 'valid')
        self.config.update_value_and_default(self.KEYS.CONFIG.METHOD, method, 'mean')
        self.model = None
        self.name = name

    def kernel(self, xs):
        return self.model(xs)

    def build(self, xs):
        if isinstance(xs, tf.Tensor):
            if self.config[self.KEYS.CONFIG.METHOD] == 'mean':
                self.model = tf.layers.AveragePooling2D(self.config[self.KEYS.CONFIG.POOL_SIZE],
                                                        self.config[self.KEYS.CONFIG.STRIDE],
                                                        self.config[self.KEYS.CONFIG.PADDING],
                                                        name=self.name)
            if self.config[self.KEYS.CONFIG.METHOD] == 'max':
                self.model = tf.layers.MaxPooling2D(self.config[self.KEYS.CONFIG.POOL_SIZE],
                                                    self.config[self.KEYS.CONFIG.STRIDE],
                                                    self.config[self.KEYS.CONFIG.PADDING],
                                                    name=self.name)
            return
        raise NotImplementedError('not supported the type of {}'.format(type(xs)))

    @property
    def parameters(self):
        return []


class UpSampling2D(Model):
    class KEYS(Model.KEYS):
        class CONFIG:
            SIZE = 'size'
            IS_SCALE = 'is_scale'
            METHOD = 'method'
            ALIGN_CORNERS = 'align_corners'

    def __init__(self,
                 name,
                 size=None,
                 is_scale=None,
                 method=None,
                 align_corners=None):
        super().__init__(name)
        self.config.update(self.KEYS.CONFIG.SIZE, size)
        self.config.update_value_and_default(self.KEYS.CONFIG.IS_SCALE, is_scale, True)
        self.config.update_value_and_default(self.KEYS.CONFIG.METHOD, method, 0)
        self.config.update_value_and_default(self.KEYS.CONFIG.ALIGN_CORNERS, align_corners, False)

    def kernel(self, xs):
        tag_size = self.config[self.KEYS.CONFIG.SIZE]
        if len(xs.shape) == 3:
            if self.config[self.KEYS.CONFIG.IS_SCALE]:
                size_h = tag_size[0] * int(xs.shape[0])
                size_w = tag_size[1] * int(xs.shape[1])
                tag_size = [int(size_h), int(size_w)]
        elif len(xs.shape) == 4:
            if self.config[self.KEYS.CONFIG.IS_SCALE]:
                size_h = tag_size[0] * int(xs.shape[1])
                size_w = tag_size[1] * int(xs.shape[2])
                tag_size = [int(size_h), int(size_w)]
        else:
            raise Exception("Do not support shape {}".format(xs.shape))
        return tf.image.resize_images(
            images=xs,
            size=tag_size,
            method=self.config[self.KEYS.CONFIG.METHOD],
            align_corners=self.config[self.KEYS.CONFIG.ALIGN_CORNERS])

    def build(self, x):
        pass

    def parameters(self):
        return []
