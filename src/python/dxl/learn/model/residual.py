import tensorflow as tf 
import numpy as np 
from dxl.learn.core import Model
from dxl.learn.model.cnn import StackedConv2D, InceptionBlock

class Residual(Model):
    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            RATIO = 'ratio'

        class GRAPH(Model.KEYS.GRAPH):
            SHORT_CUT = 'short_cut'

    def __init__(self, info, inputs, short_cut, ratio, config=None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            graphs={self.KEYS.GRAPHS.SHORT_CUT: short_cut},
            config=self._parse_input_config(config, {
                self.KEYS.CONFIG.RATIO: ratio
            })
        )

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.RATIO: 0.3}

    def _parse_input_config(self, config, **kwargs):
        if config is None:
            config = {}
        
        return config.update(kwargs)

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        sub_graph = self.graphs[self.KEYS.GRAPH.SHORT_CUT]
        h = sub_graph(x)
        with tf.name_scope("add"):
            x = x + h * self.config(self.KEYS.CONFIG.RATIO)
        return x


# ============================================================================
#                       Special Residual
# ============================================================================
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

