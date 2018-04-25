import tensorflow as tf
from ..core import Model, GraphInfo, Tensor
from dxl.fs import Path


class Summation(Model):
    def __init__(self, name: Path, graph_info: GraphInfo):
        super().__init__(name, inputs=None, submodels=None, graph_info=graph_info)

    def kernel(self, inputs=None):
        if inputs is None or len(inputs) == 0:
            return None
        else:
            tf_tensors = []
            if isinstance(inputs, dict):
                for v in inputs.values():
                    tf_tensors.append(self.tensorflow_tensor(v))
            else:
                for v in inputs:
                    tf_tensors.append(self.tensorflow_tensor(v))
            result = tf.add_n(tf_tensors)
            return Tensor(result, None, self.graph_info.update(name=result.name))
