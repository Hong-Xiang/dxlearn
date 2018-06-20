import tensorflow as tf 
import numpy as np
from dxl.learn.core import Model 
from dxl.learn.model.cnn import Conv2D, InceptionBlock

__all__ = [
    'Stack'
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
