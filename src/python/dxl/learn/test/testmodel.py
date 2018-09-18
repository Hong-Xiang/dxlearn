import tensorflow as tf
# from dxl.learn.core import Model
from dxl.learn.model.base import Model

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

    def __init__(self, name='UnitBlock'):
        super().__init__(name)
        self.inputs = []
    def kernel(self, inputs):
        self.inputs.append(inputs)
        return inputs
