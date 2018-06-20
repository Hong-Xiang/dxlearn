# -*- coding: utf-8 -*-

import tensorflow as tf
from fs import path as fp
from ...core import Model, Tensor
from .. import activation

__all__ = [
    # 'Conv1D',
    'Conv2D',
    'InceptionBlock',
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

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            FILTERS = 'filters'
            KERNEL_SIZE = 'kernel_size'
            STRIDES = 'strides'
            PADDING = 'padding'
            ACTIVATION = 'activation'


    def __init__(
            self,
            info='conv2d',
            inputs=None,
            filters=None,
            kernel_size=None,
            strides=None,
            padding=None,
            activation=None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            config={
                self.KEYS.CONFIG.FILTERS: filters,
                self.KEYS.CONFIG.KERNEL_SIZE: kernel_size,
                self.KEYS.CONFIG.STRIDES: strides,
                self.KEYS.CONFIG.PADDING: padding,
                self.KEYS.CONFIG.ACTIVATION: activation
            })

    @classmethod
    def _default_config(cls):
        return {
            cls.KEYS.CONFIG.STRIDES: (1, 1),
            cls.KEYS.CONFIG.PADDING: 'same',
            cls.KEYS.CONFIG.ACTIVATION: 'none'
        }

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        acc = activation.unified_config(
            self.config(self.KEYS.CONFIG.ACTIVATION))
        x = activation.apply(acc, x, 'pre')
        x = tf.layers.conv2d(
            inputs=x,
            filters=self.config(self.KEYS.CONFIG.FILTERS),
            kernel_size=self.config(self.KEYS.CONFIG.KERNEL_SIZE),
            strides=self.config(self.KEYS.CONFIG.STRIDES),
            padding=self.config(self.KEYS.CONFIG.PADDING),
            name='convolution',
            reuse=self._created)
        x = activation.apply(acc, x, 'post')
        return x
        

class InceptionBlock(Model):
    """InceptionBlock model
    Arguments:
        name: Path := dxl.fs.
        inputs: Tensor input.
        paths: Integer.
        activation: Activation function. Set it to None to maintain a linear activation.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            PATHS = 'paths'
            ACTIVATION = 'activation'
            FILTERS = 'filters'
            KERNEL_SIZE = 'kernel_size'
            ACTIVATION = 'activation'

    def __init__(
            self,
            info='incept',
            inputs=None,
            paths=None,
            activation=None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            config={
                self.KEYS.CONFIG.PATHS: paths,
                self.KEYS.CONFIG.ACTIVATION: activation
            })

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.PATHS: 2, cls.KEYS.CONFIG.ACTIVATION: 'linear'}

    def _short_cut(self, name, inputs, config):
        shortcut = Conv2D(
            info=self.info.child_scope(name),
            inputs=inputs,
            filters=config[self.KEYS.CONFIG.FILTERS],
            kernel_size=config[self.KEYS.CONFIG.KERNEL_SIZE],
            activation=config[self.KEYS.CONFIG.ACTIVATION])
        return shortcut

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        filters = x.shape.as_list()[-1]
        acc = activation.unified_config(
            self.config(self.KEYS.CONFIG.ACTIVATION))
        x = activation.apply(acc, x, 'pre')
        paths = []
        for i_path in range(self.config(self.KEYS.CONFIG.PATHS)):
            config = {
                self.KEYS.CONFIG.FILTERS: filters,
                self.KEYS.CONFIG.KERNEL_SIZE: 1,
                self.KEYS.CONFIG.ACTIVATION: 'linear'
            }
            key = 'conv_{}'.format(i_path)
            h = self.get_or_create_graph(key, self._short_cut(key, x, config))()
            for j in range(i_path):
                config = {
                    self.KEYS.CONFIG.FILTERS: filters,
                    self.KEYS.CONFIG.KERNEL_SIZE: 3,
                    self.KEYS.CONFIG.ACTIVATION: 'pre'
                }
                key = 'conv2d_{}'.format(j+1)
                h = self.get_or_create_graph(key, self._short_cut(key, x, config))()
            paths.append(h)
        with tf.name_scope('concat'):
            x = tf.concat(paths, axis=-1)

        config = {
            self.KEYS.CONFIG.FILTERS: filters,
            self.KEYS.CONFIG.KERNEL_SIZE: 1,
            self.KEYS.CONFIG.ACTIVATION: 'pre'
        }
        key = 'conv_end'
        x = self.get_or_create_graph(key, self._short_cut(key, x, config))()
        return x


class DownSampling2D(Model):
    """DownSampling2D Block
    Arguments:
        name: Path := dxl.fs.
            A unique block name.
        inputs: 4-D Tensor in the shape of (batch, height, width, channels) or 3-D Tensor in the shape of (height, width, channels).
        size: tuple of int/float
            (height, width) scale factor or new size of height and width.
        is_scale: boolean
            If True (default), the `size` is the scale factor; otherwise, the `size` are numbers of pixels of height and width.
        method: int
            - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
            - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
            - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
            - Index 3 ResizeMethod.AREA, Area interpolation.
        align_corners: boolean
            If True, exactly align all 4 corners of the input and output. Default is False.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            SIZE = 'size'
            IS_SCALE = 'is_scale'
            METHOD = 'method'
            ALIGN_CORNERS = 'align_corners'

    def __init__(self,
                 info='downsample2d',
                 inputs=None,
                 size=None,
                 is_scale=None,
                 method=None,
                 align_corners=None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            config={
                self.KEYS.CONFIG.SIZE: size,
                self.KEYS.CONFIG.IS_SCALE: is_scale,
                self.KEYS.CONFIG.METHOD: method,
                self.KEYS.CONFIG.ALIGN_CORNERS: align_corners
            })

    @classmethod
    def _default_config(cls):
        return {
            cls.KEYS.CONFIG.IS_SCALE: True,
            cls.KEYS.CONFIG.METHOD: 0,
            cls.KEYS.CONFIG.ALIGN_CORNERS: False
        }

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        x_shape = x.shape
        tag_size = self.config(self.KEYS.CONFIG.SIZE)
        if len(x_shape) == 3:
            if self.config(self.KEYS.CONFIG.IS_SCALE):
                ratio_size = self.config(self.KEYS.CONFIG.SIZE)
                size_h = ratio_size[0] * int(x_shape[0])
                size_w = ratio_size[1] * int(x_shape[1])
                tag_size = [int(size_h), int(size_w)]
        elif len(x_shape) == 4:
            if self.config(self.KEYS.CONFIG.IS_SCALE):
                ratio_size = self.config(self.KEYS.CONFIG.SIZE)
                size_h = ratio_size[0] * int(x_shape[1])
                size_w = ratio_size[1] * int(x_shape[2])
                tag_size = [int(size_h), int(size_w)]
        else:
            raise Exception("Do not support shape {}".format(x_shape))
        with tf.name_scope('downsampling'):
            h = tf.image.resize_images(
                images=x,
                size=tag_size,
                method=self.config(self.KEYS.CONFIG.METHOD),
                align_corners=self.config(self.KEYS.CONFIG.ALIGN_CORNERS))

        return h


class UpSampling2D(Model):
    """UpSampling2D block
    Arguments:
        Arguments:
        name: Path := dxl.fs.
            A unique block name.
        inputs: 4-D Tensor in the shape of (batch, height, width, channels) or 3-D Tensor in the shape of (height, width, channels).
        size: tuple of int/float
            (height, width) scale factor or new size of height and width.
        is_scale: boolean
            If True (default), the `size` is the scale factor; otherwise, the `size` are numbers of pixels of height and width.
        method: int
            - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
            - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
            - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
            - Index 3 ResizeMethod.AREA, Area interpolation.
        align_corners: boolean
            If True, align the corners of the input and output. Default is False.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            SIZE = 'size'
            IS_SCALE = 'is_scale'
            METHOD = 'method'
            ALIGN_CORNERS = 'align_corners'

    def __init__(self,
                 info='upsample2d',
                 inputs=None,
                 size=None,
                 is_scale=None,
                 method=None,
                 align_corners=None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            config={
                self.KEYS.CONFIG.SIZE: size,
                self.KEYS.CONFIG.IS_SCALE: is_scale,
                self.KEYS.CONFIG.METHOD: method,
                self.KEYS.CONFIG.ALIGN_CORNERS: align_corners
            })

    @classmethod
    def _default_config(cls):
        return {
            cls.KEYS.CONFIG.IS_SCALE: True,
            cls.KEYS.CONFIG.METHOD: 0,
            cls.KEYS.CONFIG.ALIGN_CORNERS: False
        }

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        x_shape = x.shape
        tag_size = self.config(self.KEYS.CONFIG.SIZE)
        if len(x_shape) == 3:
            if self.config(self.KEYS.CONFIG.IS_SCALE):
                ratio_size = self.config(self.KEYS.CONFIG.SIZE)
                size_h = ratio_size[0] * int(x_shape[0])
                size_w = ratio_size[1] * int(x_shape[1])
                tag_size = [int(size_h), int(size_w)]
        elif len(x_shape) == 4:
            if self.config(self.KEYS.CONFIG.IS_SCALE):
                ratio_size = self.config(self.KEYS.CONFIG.SIZE)
                size_h = ratio_size[0] * int(x_shape[1])
                size_w = ratio_size[1] * int(x_shape[2])
                tag_size = [int(size_h), int(size_w)]
        else:
            raise Exception("Do not support shape {}".format(x_shape))
        with tf.name_scope('upsampling'):
            h = tf.image.resize_images(
                images=x,
                size=tag_size,
                method=self.config(self.KEYS.CONFIG.METHOD),
                align_corners=self.config(self.KEYS.CONFIG.ALIGN_CORNERS))

        return h
