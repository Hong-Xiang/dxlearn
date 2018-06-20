import tensorflow as tf 
import numpy as np
from dxl.learn.core import Model 
from dxl.learn.model.cnn import Conv2D, InceptionBlock
from .residual import ResidualIncept, ResidualStackedConv

__all__ = [
    'Stack', 
    'StackedResidualIncept',
    'StackedResidualConv'
]

class Stack(Model):
    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            NB_LAYERS = 'nb_layers'

        class GRAPHS(Model.KEYS.GRAPH):
            SHORT_CUT = 'short_cut'

    def __init__(self, info, inputs, short_cut, nb_layers, config=None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            graphs={self.KEYS.GRAPHS.SHORT_CUT: short_cut},
            config=self._parse_input_config(config, {
                self.KEYS.CONFIG.NB_LAYERS: nb_layers
            })
        )

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.NB_LAYERS: 2}

    def _parse_input_config(self, config, **kwargs):
        if config is None:
            config = {}
        return config.update(kwargs)

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        sub_graph = self.graphs[self.KEYS.GRAPH.SHORT_CUT]
        for _ in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            x = sub_graph(x)
        return x


# ============================================================================
#                          Special Stack
# ============================================================================
class StackedConv2D(Model):
    """StackedConv2D convolution model
    Arguments:
        name: Path := dxl.fs.
        inputs: Tensor input.
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

    def __init__(
            self,
            info,
            inputs=None,
            nb_layers=None,
            filters=None,
            kernel_size=None,
            strides=None,
            padding=None,
            activation=None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            config={
                self.KEYS.CONFIG.NB_LAYERS: nb_layers,
                self.KEYS.CONFIG.FILTERS: filters,
                self.KEYS.CONFIG.KERNEL_SIZE: kernel_size,
                self.KEYS.CONFIG.STRIDES: strides,
                self.KEYS.CONFIG.PADDING: padding,
                self.KEYS.CONFIG.ACTIVATION: activation
            })

    @classmethod
    def _default_config(cls):
        return {
            cls.KEYS.CONFIG.NB_LAYERS: 2,
            cls.KEYS.CONFIG.FILTERS: 5,
            cls.KEYS.CONFIG.STRIDES: (1, 1),
            cls.KEYS.CONFIG.PADDING: 'valid',
            cls.KEYS.CONFIG.ACTIVATION: 'linear'
        }

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        for i in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            x = Conv2D(
                'conv2d_{}'.format(i),
                inputs=x,
                filters=self.config(self.KEYS.CONFIG.FILTERS),
                kernel_size=self.config(self.KEYS.CONFIG.KERNEL_SIZE),
                strides=self.config(self.KEYS.CONFIG.STRIDES),
                padding=self.config(self.KEYS.CONFIG.PADDING),
                activation=self.config(self.KEYS.CONFIG.ACTIVATION))()
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
            key = self.KEYS.GRAPHS.SHORT_CUT
            sub_graph = self.get_or_create_graph(key,
                                                 self._short_cut(key, x))
            x = sub_graph(x)
        return x
