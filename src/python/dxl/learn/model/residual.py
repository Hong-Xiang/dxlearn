import tensorflow as tf 
import numpy as np 
from dxl.learn.core import Model
from dxl.learn.model.cnn import InceptionBlock


class Residual(Model):
    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            RATIO = 'ratio'

        class GRAPHS(Model.KEYS.GRAPH):
            SHORT_CUT = 'short_cut'

    def __init__(self, info, inputs=None, short_cut=None, ratio=None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            graphs={self.KEYS.GRAPHS.SHORT_CUT: short_cut},
            config={self.KEYS.CONFIG.RATIO: ratio})

    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.RATIO: 0.3}

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        h = self.graphs[self.KEYS.GRAPHS.SHORT_CUT](x)
        with tf.name_scope("add"):
            x = x + h * self.config(self.KEYS.CONFIG.RATIO)
        return x