import tensorflow as tf
from dxl.learn.core import Model


class UnitBlock(Model):
    """UnitBlock block for test use.
    Arguments:
        name: Path := dxl.fs.
        inputs: Tensor input.
        graph_info: GraphInfo or DistributeGraphInfo
    Return:
        inputs
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass

        class CONFIG:
            pass

    def __init__(self, info='UnitBlock', inputs=None):
        super().__init__(info, tensors={self.KEYS.TENSOR.INPUT: inputs})

    def kernel(self, inputs):
        x = inputs[self.KEYS.TENSOR.INPUT]
        return x
