# -*- coding: utf-8 -*-

import tensorflow as tf
from fs import path as fp
from ...core import Model, Tensor
from .. import activation

__all__ = [
    # 'Conv1D',
    'Conv2D',
    'StackedConv2D',
    'InceptionBlock',
    'UnitBlock',
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
        input_tensor: Tensor input.
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

    @classmethod
    def default_config(cls):
        return {
            cls.KEYS.CONFIG.PADDING: 'same',
            cls.KEYS.CONFIG.STRIDES: (1, 1),
            cls.KEYS.CONFIG.ACTIVATION: 'none',
        }

    def __init__(self,
                 name='conv2d',
                 input_tensor=None,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 padding=None,
                 activation=None,
                 graph_info=None):
        super().__init__(
            name,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.FILTERS: filters,
                self.KEYS.CONFIG.KERNEL_SIZE: kernel_size,
                self.KEYS.CONFIG.STRIDES: strides,
                self.KEYS.CONFIG.PADDING: padding,
                self.KEYS.CONFIG.ACTIVATION: activation
            })

    @classmethod
    def default_config(cls):
        return {
            cls.KEYS.CONFIG.FILTERS: 5,
            cls.KEYS.CONFIG.STRIDES: (1,1),
            cls.KEYS.CONFIG.PADDING: 'valid',
            cls.KEYS.CONFIG.ACTIVATION: 'linear'}

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
            name='convolution')
        x = activation.apply(acc, x, 'post')
        return x


class StackedConv2D(Model):
    """StackedConv2D convolution model
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        nb_layers: Integer, the number of stacked layers.
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
            NB_LAYERS = 'nb_layers'
            FILTERS = 'filters'
            KERNEL_SIZE = 'kernel_size'
            STRIDES = 'strides'
            PADDING = 'padding'
            ACTIVATION = 'activation'

    def __init__(self,
                 name,
                 input_tensor=None,
                 nb_layers=None,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 padding=None,
                 activation=None,
                 graph_info=None):
        super().__init__(
            name,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.NB_LAYERS: nb_layers,
                self.KEYS.CONFIG.FILTERS: filters,
                self.KEYS.CONFIG.KERNEL_SIZE: kernel_size,
                self.KEYS.CONFIG.STRIDES: strides,
                self.KEYS.CONFIG.PADDING: padding,
                self.KEYS.CONFIG.ACTIVATION: activation
            })

    @classmethod
    def default_config(cls):
        return {
            cls.KEYS.CONFIG.NB_LAYERS: 2,
            cls.KEYS.CONFIG.FILTERS: 5,
            cls.KEYS.CONFIG.STRIDES: (1,1),
            cls.KEYS.CONFIG.PADDING: 'valid',
            cls.KEYS.CONFIG.ACTIVATION: 'linear'}

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        for i in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            x = Conv2D(
                name='conv2d_{}'.format(i),
                input_tensor=x,
                filters=self.config(self.KEYS.CONFIG.FILTERS),
                kernel_size=self.config(self.KEYS.CONFIG.KERNEL_SIZE),
                strides=self.config(self.KEYS.CONFIG.STRIDES),
                padding=self.config(self.KEYS.CONFIG.PADDING),
                activation=self.config(self.KEYS.CONFIG.ACTIVATION))()
        return x


class InceptionBlock(Model):
    """InceptionBlock model
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
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

    def __init__(self,
                 name='incept',
                 input_tensor=None,
                 paths=None,
                 activation=None,
                 graph_info=None):
        super().__init__(
            name,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.PATHS: paths,
                self.KEYS.CONFIG.ACTIVATION: activation
            })

    @classmethod
    def default_config(cls):
        return {
            cls.KEYS.CONFIG.PATHS: 2,
            cls.KEYS.CONFIG.ACTIVATION: 'linear'}

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        filters = x.shape.as_list()[-1]
        acc = activation.unified_config(
            self.config(self.KEYS.CONFIG.ACTIVATION))
        x = activation.apply(acc, x, 'pre')
        paths = []
        for i_path in range(self.config(self.KEYS.CONFIG.PATHS)):
            with tf.variable_scope('path_{}'.format(i_path)):
                h = Conv2D(
                    name='conv_0',
                    input_tensor=x,
                    filters=filters,
                    kernel_size=1,
                    strides=(1, 1),
                    padding='same',
                    activation='linear')()
                for j in range(i_path):
                    h = Conv2D(
                        name='conv2d_{}'.format(j + 1),
                        input_tensor=h,
                        filters=filters,
                        kernel_size=3,
                        strides=(1, 1),
                        padding='same',
                        activation='pre')()
                paths.append(h)
        with tf.name_scope('concat'):
            x = tf.concat(paths, axis=-1)
        x = Conv2D(
            name='conv_end',
            input_tensor=x,
            filters=filters,
            kernel_size=1,
            strides=(1, 1),
            padding='same',
            activation='pre')()
        return x


class UnitBlock(Model):
    """UnitBlock block for test use.
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        graph_info: GraphInfo or DistributeGraphInfo
    Return:
        input_tensor
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            pass

    def __init__(self, name='UnitBlock', input_tensor=None, graph_info=None):
        super().__init__(
            name,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            graph_info=graph_info)

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        return x


class DownSampling2D(Model):
    """DownSampling2D Block
    Arguments:
        name: Path := dxl.fs.
            A unique block name.
        input_tensor: 4-D Tensor in the shape of (batch, height, width, channels) or 3-D Tensor in the shape of (height, width, channels).
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
                 name='downsample2d',
                 input_tensor=None,
                 size=None,
                 is_scale=True,
                 method=0,
                 align_corners=False,
                 graph_info=None):
        super().__init__(
            name,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.SIZE: size,
                self.KEYS.CONFIG.IS_SCALE: is_scale,
                self.KEYS.CONFIG.METHOD: method,
                self.KEYS.CONFIG.ALIGN_CORNERS: align_corners
            })

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
            raise Exception("Donot support shape {}".format(x_shape))
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
        input_tensor: 4-D Tensor in the shape of (batch, height, width, channels) or 3-D Tensor in the shape of (height, width, channels).
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
                 name='upsample2d',
                 input_tensor=None,
                 size=None,
                 is_scale=True,
                 method=0,
                 align_corners=False,
                 graph_info=None):
        super().__init__(
            name,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.SIZE: size,
                self.KEYS.CONFIG.IS_SCALE: is_scale,
                self.KEYS.CONFIG.METHOD: method,
                self.KEYS.CONFIG.ALIGN_CORNERS: align_corners
            })

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
            raise Exception("Donot support shape {}".format(x_shape))
        with tf.name_scope('upsampling'):
            h = tf.image.resize_images(
                images=x,
                size=tag_size,
                method=self.config(self.KEYS.CONFIG.METHOD),
                align_corners=self.config(self.KEYS.CONFIG.ALIGN_CORNERS))

        return h
