# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from fs import path as fp
from .blocks import Conv2D, StackedConv2D, InceptionBlock, UnitBlock
from ...core import Model
from ...core import Tensor

__all__ = [
    'ResidualIncept', 'ResidualStackedConv', 'StackedResidualIncept',
    'StackedResidualConv'
]


class Residual(Model):
    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            RATIO = 'ratio'

        class GRAPH(Model.KEYS.GRAPH):
            SHORT_CUT = 'short_cut'

    def __init__(self, info, tensor, short_cut, ratio, config):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: tensor},
            graphs={self.KEYS.GRAPHS.SHORT_CUT: short_cut},
            config=self._parse_input_config(config, {
                self.KEYS.CONFIG.RATIO: ratio
            })
        )


class ResidualIncept(Model):
    """ResidualIncept Block
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        ratio: The decimal.
        sub_block: InceptionBlock instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            RATIO = 'ratio'

        class SUB_BLOCK:
            NAME = 'InceptionBlock'

    def __init__(self,
                 info,
                 input=None,
                 ratio=None,
                 sub_block: InceptionBlock = None):
        super().__init__(
            info,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            submodels={self.KEYS.SUB_BLOCK.NAME: sub_block},
            config={
                self.KEYS.CONFIG.RATIO: ratio,
            })

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.RATIO: 0.3}

    def sub_block_maker(self, name, input_tensor):
        sub_block = InceptionBlock(
            self.info.child_scope(name),
            input_tensor=input_tensor,
            paths=3,
            activation='incept')

        return sub_block

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        key = self.KEYS.SUB_BLOCK.NAME
        sub_block = self.get_or_create_graph(key, self.sub_block_maker(key, x))
        h = sub_block(x)
        with tf.name_scope('add'):
            x = x + h * self.config(self.KEYS.CONFIG.RATIO)
        return x


class ResidualStackedConv(Model):
    """ ResidualStackedConv Block
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        ratio: The decimal.
        sub_block: StackedConv2D instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            RATIO = 'ratio'

        class SUB_BLOCK:
            NAME = 'StackedConv2DBlock'

    def __init__(self,
                 info,
                 input_tensor=None,
                 ratio=None,
                 sub_block: StackedConv2D = None):
        super().__init__(
            info,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            submodels={self.KEYS.SUB_BLOCK.NAME: sub_block},
            config={self.KEYS.CONFIG.RATIO: ratio})

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.RATIO: 0.1}

    def sub_block_maker(self, name, input_tensor):
        return StackedConv2D(
            self.info.child_scope(name),
            input_tensor=input_tensor,
            nb_layers=2,
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='basic')

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        key = self.KEYS.SUB_BLOCK.NAME
        sub_block = self.get_or_create_graph(key, self.sub_block_maker(key, x))
        h = sub_block(x)
        with tf.name_scope('add'):
            x = x + h * self.config(self.KEYS.CONFIG.RATIO)
        return x


class StackedResidualIncept(Model):
    """StackedResidual Block
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        nb_layers: Integer.
        sub_block: ResidualIncept Instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            NB_LAYERS = 'nb_layers'

        class SUB_BLOCK:
            NAME = 'ResidualIncept'

    def __init__(self,
                 info,
                 input_tensor=None,
                 nb_layers=None,
                 sub_block: ResidualIncept = None):
        super().__init__(
            info,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            submodels={self.KEYS.SUB_BLOCK.NAME: sub_block},
            config={self.KEYS.CONFIG.NB_LAYERS: nb_layers})

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.NB_LAYERS: 2}

    def sub_block_maker(self, name, input_tensor):
        return ResidualIncept(
            self.info.child_scope(name), input_tensor=input_tensor, ratio=0.3)

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        key = self.KEYS.SUB_BLOCK.NAME
        for i in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            sub_block = self.get_or_create_graph(key,
                                                 self.sub_block_maker(key, x))
            x = sub_block(x)
        return x


class StackedResidualConv(Model):
    """StackedResidual Block
    Arguments:
        name: Path := dxl.fs.
        input_tensor: Tensor input.
        nb_layers: Integer.
        sub_block: ResidualStackedConv Instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            NB_LAYERS = 'nb_layers'

        class SUB_BLOCK:
            NAME = 'ResidualStackedConv'

    def __init__(self,
                 info,
                 input_tensor=None,
                 nb_layers=None,
                 sub_block: ResidualStackedConv = None):
        super().__init__(
            info,
            inputs={self.KEYS.TENSOR.INPUT: input_tensor},
            submodels={self.KEYS.SUB_BLOCK.NAME: sub_block},
            config={self.KEYS.CONFIG.NB_LAYERS: nb_layers})

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.NB_LAYERS: 2}

    def sub_block_maker(self, name, input_tensor):
        return ResidualStackedConv(
            self.info.child_scope(name), input_tensor=input_tensor, ratio=0.1)

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        for i in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            key = "{}_{}".format(self.KEYS.SUB_BLOCK.NAME, i)
            sub_block = self.get_or_create_graph(key,
                                                 self.sub_block_maker(key, x))
            x = sub_block(x)
        return x
