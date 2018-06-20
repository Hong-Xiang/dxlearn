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
            MODELS = 'models'

    def __init__(self, info, inputs=None, models=None, nb_layers=None):
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            graphs={self.KEYS.GRAPHS.MODELS: models},
            config={self.KEYS.CONFIG.NB_LAYERS: nb_layers}
        )
    
    @classmethod
    def _default_config(cls):
        return {cls.KEYS.CONFIG.NB_LAYERS: 2}

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        for _ in range(self.config(self.KEYS.CONFIG.NB_LAYERS)):
            x = self.graphs[self.KEYS.GRAPHS.MODELS](x)
        return x
