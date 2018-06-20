import tensorflow as tf 
import numpy as np 
from dxl.learn.core import Model
from dxl.learn.model.cnn import InceptionBlock


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