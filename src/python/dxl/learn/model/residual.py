import tensorflow as tf 
import numpy as np 
from dxl.core import Model


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
