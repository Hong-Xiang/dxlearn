import tensorflow as tf 
import numpy as np
from dxl.core import Model 


class Stack(Model):
    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            NB_LAYERS = 'nb_layers'

        class GRAPHS(Model.KEYS.GRAPH):
            SHORT_CUT = 'short_cut'

    def __init__(self, info, inputs, short_cut, nb_layers, config):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: tensor},
            graphs={self.KEYS.GRAPHS.SHORT_CUT: short_cut},
            config=self._parse_input_config(config, {
                self.KEYS.CONFIG.NB_LAYERS: nb_layers
            })
        )