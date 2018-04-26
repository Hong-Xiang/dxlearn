from .graph import Graph
from .tensor import Tensor
from .distribute import Host
from .graph_info import GraphInfo, DistributeGraphInfo
from typing import Dict
from dxl.fs import Path
import tensorflow as tf


class Model(Graph):
    """
    A special case of Graph, which all inputs are listed in inputs, i.e. no Tensor
    created in constructing model will introduce external information, works like a
    function. Note `Model` is not "pure" function since there maybe variables
    for model itself.  

    Model provide `__call__` method, which make reuse of Model much more easier.
    """

    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            INPUT = 'input'

    def __init__(self,
                 name: Path,
                 inputs: Dict[str, Tensor] = None,
                 submodels: Dict[str, 'Model'] = None,
                 graph_info: GraphInfo = None,
                 config: Dict[str, 'Config'] = None):
        super().__init__(
            name,
            tensors=inputs,
            subgraphs=submodels,
            graph_info=graph_info,
            config=config)
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
        with self.graph_info.variable_scope(reuse=not is_create):
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
        if isinstance(inputs, (Tensor, tf.Tensor)):
            inputs = {self.KEYS.TENSOR.INPUT: inputs}
        if inputs is not None:
            if isinstance(inputs, dict):
                for k in self.inputs:
                    if not k in inputs:
                        inputs[k] = self.inputs[k]
        return inputs

    def pre_kernel_in_scope(self, inputs, is_create):
        return inputs

    def post_kernel_in_scope(self, results, is_create):
        return results

    def post_kernel(self, results, is_create):
        if is_create:
            if results is None:
                results = {}
        if results is None:
            return results
        if isinstance(results, (Tensor, tf.Tensor)):
            results = {self.KEYS.TENSOR.MAIN: results}
        if is_create:
            for k, v in results.items():
                self.outputs[k] = v
        if len(results) == 1 and self.KEYS.TENSOR.MAIN in results:
            return results[self.KEYS.TENSOR.MAIN]
        return results
