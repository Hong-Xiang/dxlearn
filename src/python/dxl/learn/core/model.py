from .graph import Graph
from .tensor import Tensor
from .distribute import Host
from .graph_info import GraphInfo, DistributeGraphInfo
from typing import Dict
from dxl.fs import Path


class Model(Graph):
    """
    A special case of Graph, which all inputs are listed in inputs, i.e. no Tensor
    created in constructing model will introduce external information, works like a
    function. Note `Model` is not "pure" function since there maybe variables
    for model itself.  

    Model provide `__call__` method, which make reuse of Model much more easier.
    """

    def __init__(self, name: Path, inputs: Dict[str, Tensor]=None,
                 submodels: Dict[str, 'Model']=None,
                 graph_info: GraphInfo=None):
        super().__init__(name, tensors=inputs, subgraphs=submodels, graph_info=graph_info)
        self.inputs = {}
        self.outputs = {}
        self.construct(inputs, True)

    def __call__(self, inputs=None):
        """
        Returns:
            A dict of tensors.
        """
        return self.construct(inputs, False)

    def construct(self, inputs, is_create):
        if inputs is None:
            inputs = {}
        inputs = self.pre_kernel(inputs, is_create)
        with self.graph_info.variable_scope():
            inputs = self.pre_kernel_in_scope(inputs, is_create)
            results = self.kernel(inputs)
            results = self.post_kernel_in_scope(results, is_create)
        return self.post_kernel(results, is_create)

    def kernel(self, inputs):
        return {}

    def pre_kernel(self, inputs, is_create):
        if is_create:
            for k, v in inputs.items():
                self.inputs[k] = v
        return inputs

    def pre_kernel_in_scope(self, inputs, is_create):
        return inputs

    def post_kernel_in_scope(self, results, is_create):
        return results

    def post_kernel(self, results, is_create):
        if is_create:
            if results is None:
                results = {}
            if isinstance(results, Tensor):
                results = {self.KEYS.TENSOR.MAIN: results}
            for k, v in results.items():
                self.outputs[k] = v
        return results
