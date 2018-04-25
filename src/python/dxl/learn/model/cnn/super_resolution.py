import tensorflow as tf
import numpy as np 
from ...core import Model 
from .blocks import StackedConv2D 
from .residual import StackedResidualConv, StackedResidualIncept 


__all__ = [
    "SuperResolution2x",
    "SuperResolutionBlock",
    "SuperResolutionMultiScale",
    "SuperResolutionMultiScalev2",
]


class SuperResolution2x(Model):
    """ SuperResolution2x Block
    Arguments:

    """
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass
        class CONFIG:
            NB_LAYERS = 'nb_layers'
            FILTERS = 'filters'
            BOUNDARY_CROP = 'boundary_crop'
            SUB_BLOCK = 'sub_block'

    def __init__(self, name,
                    inputs: Dict[str, Tensor]=None, 
                    nb_layers: int=None,
                    filters: int=None,
                    boundary_crop=None,
                    sub_block: (StackedConv2D, StackedResidualConv, StackedResidualIncept)=None,
                    graph_info=None):
    super().__init__(
        name,
        inputs=inputs,
        graph_info=graph_info,
        config={
            self.KEYS.CONFIG.NB_LAYERS: nb_layers,
            self.KEYS.CONFIG.FILTERS: filters,
            self.KEYS.CONFIG.BOUNDARY_CROP: boundary_crop,
            self.KEYS.CONFIG.SUB_BLOCK: sub_block
        })
        
    def pre_kernel(self, inputs, is_create):
        sub_block = self.config(self.KEYS.CONFIG.SUB_BLOCK)
        if not isinstance(
            sub_block, 
            (StackedConv2D, StackedResidualConv, StackedResidualIncept)):
            init_x = np.ones([1, 10, 10, 3], dtype="float32")
            sub_block = StackedConv2D(
                name='kernel',
                input_tensor=tf.constant(init_x),
                nb_layers=self.config(self.KEYS.CONFIG.NB_LAYERS),
                filters=self.config(self.KEYS.CONFIG.FILTERS))

        if is_create:
            for k, v in inputs.items():
                self.inputs[k] = v
        if isinstance(inputs, (Tensor, tf.Tensor)):
            inputs = {self.KEYS.TENSOR.INPUT: inputs}
        if inputs is not None:
            if isinstance(inputs, dict):
                for k in self.inputs:
                    if not k in inputs:
                        inputs[k] = self.inputs[k]
        return inputs

    def kernel(self, inputs):
        