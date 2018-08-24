from .base import ConfigurableFunction

from enum import Enum

from doufo.tensor import shape


class DownSampling2D(ConfigurableFunction):
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

    class CONFIG:
        SIZE = 'size'
        IS_SCALE = 'is_scale'
        METHOD = 'method'
        ALIGN_CORNERS = 'align_corners'

    class METHODS(Enum):
        BILINEAR: 0
        NEAREST_NEIGHBOR: 1
        BICUBIC: 2

    def __init__(self, name='downsample2d', size=None, is_scale=None, method=None, align_corners=None):
        super().__init__(name),
        self._config.update({
            self.CONFIG.SIZE: size,
            self.CONFIG.IS_SCALE: is_scale if is_scale is not None else True,
            self.CONFIG.METHOD: method if method is not None else self.METHODS.BILINEAR,
            self.CONFIG.ALIGN_CORNERS: align_corners if align_corners is not None else False
        })

    def kernel(self, x):
        x_shape = shape(x)

        with tf.name_scope('downsampling'):
            h = tf.image.resize_images(
                images=x,
                size=tag_size,
                method=self.config(self.KEYS.CONFIG.METHOD),
                align_corners=self.config(self.KEYS.CONFIG.ALIGN_CORNERS))

        return h

    def target_size(self, x):
        x_shape = shape(x)
        result = list(self.CONFIG.SIZE)
        if len(result) == 3:
            if self.config[self.CONFIG.IS_SCALE]:
                ratio_size = self.config(self.CONFIG.SIZE)
                size_h = ratio_size[0] * int(x_shape[0])
                size_w = ratio_size[1] * int(x_shape[1])
                result = [int(size_h), int(size_w)]
        elif len(result) == 4:
            if self.config[self.CONFIG.IS_SCALE]:
                ratio_size = self.config(self.CONFIG.SIZE)
                size_h = ratio_size[0] * int(x_shape[1])
                size_w = ratio_size[1] * int(x_shape[2])
                result = [int(size_h), int(size_w)]
        else:
            raise Exception("Do not support shape {}".format(x_shape))
        return result



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
