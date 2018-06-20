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

class ResidualIncept(Model):
    """ResidualIncept Block
    Arguments:
        name: Path := dxl.fs.
        inputs: Tensor input.
        ratio: The decimal.
        sub_graph: InceptionBlock instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            RATIO = 'ratio'

        class GRAPHS:
            SHORT_CUT = 'InceptionBlock'

    def __init__(self,
                 info,
                 inputs=None,
                 ratio=None,
                 graph: InceptionBlock = None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            graphs={self.KEYS.GRAPHS.SHORT_CUT: graph},
            config={
                self.KEYS.CONFIG.RATIO: ratio,
            })

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.RATIO: 0.3}

    def _short_cut(self, name, inputs):
        return InceptionBlock(
            self.info.child_scope(name),
            inputs=inputs,
            paths=3,
            activation='incept')

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        key = self.KEYS.GRAPHS.SHORT_CUT
        sub_graph = self.get_or_create_graph(key, self._short_cut(key, x))
        h = sub_graph(x)
        with tf.name_scope('add'):
            x = x + h * self.config(self.KEYS.CONFIG.RATIO)
        return x


class ResidualStackedConv(Model):
    """ ResidualStackedConv Block
    Arguments:
        name: Path := dxl.fs.
        inputs: Tensor input.
        ratio: The decimal.
        sub_graph: StackedConv2D instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            RATIO = 'ratio'

        class GRAPHS:
            SHORT_CUT = 'StackedConv2DBlock'

    def __init__(self,
                 info,
                 inputs=None,
                 ratio=None,
                 graph: StackedConv2D = None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            graphs={self.KEYS.GRAPHS.SHORT_CUT: graph},
            config={self.KEYS.CONFIG.RATIO: ratio})

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.RATIO: 0.1}

    def _short_cut(self, name, inputs):
        return StackedConv2D(
            self.info.child_scope(name),
            inputs=inputs,
            nb_layers=2,
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='basic')

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        key = self.KEYS.GRAPHS.SHORT_CUT
        sub_graph = self.get_or_create_graph(key, self._short_cut(key, x))
        h = sub_graph(x)
        with tf.name_scope('add'):
            x = x + h * self.config(self.KEYS.CONFIG.RATIO)
        return x


class StackedResidualIncept(Model):
    """StackedResidual Block
    Arguments:
        name: Path := dxl.fs.
        inputs: Tensor input.
        nb_layers: Integer.
        sub_graph: ResidualIncept Instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            NB_LAYERS = 'nb_layers'

        class GRAPHS:
            SHORT_CUT = 'ResidualIncept'

    def __init__(self,
                 info,
                 inputs=None,
                 nb_layers=None,
                 graph: ResidualIncept = None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            graphs={self.KEYS.GRAPHS.SHORT_CUT: graph},
            config={self.KEYS.CONFIG.NB_LAYERS: nb_layers})

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.NB_LAYERS: 2}

    def _short_cut(self, name, inputs):
        return ResidualIncept(
            self.info.child_scope(name), inputs=inputs, ratio=0.3)

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        key = self.KEYS.GRAPHS.SHORT_CUT
        for _ in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            sub_graph = self.get_or_create_graph(key,
                                                 self._short_cut(key, x))
            x = sub_graph(x)
        return x


class StackedResidualConv(Model):
    """StackedResidual Block
    Arguments:
        name: Path := dxl.fs.
        inputs: Tensor input.
        nb_layers: Integer.
        sub_graph: ResidualStackedConv Instance.
        graph_info: GraphInfo or DistributeGraphInfo
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            NB_LAYERS = 'nb_layers'

        class GRAPHS:
            SHORT_CUT = 'ResidualStackedConv'

    def __init__(self,
                 info,
                 inputs=None,
                 nb_layers=None,
                 graph: ResidualStackedConv = None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            graphs={self.KEYS.GRAPHS.SHORT_CUT: graph},
            config={self.KEYS.CONFIG.NB_LAYERS: nb_layers})

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.NB_LAYERS: 2}

    def _short_cut(self, name, inputs):
        return ResidualStackedConv(
            self.info.child_scope(name), inputs=inputs, ratio=0.1)

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        for i in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            key = "{}_{}".format(self.KEYS.GRAPHS.SHORT_CUT, i)
            sub_graph = self.get_or_create_graph(key,
                                                 self._short_cut(key, x))
            x = sub_graph(x)
        return x
