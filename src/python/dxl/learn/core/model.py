from dxl.learn.graph import Graph
from .tensor import Tensor
from typing import Dict
import tensorflow as tf


# TODO: Add self.variables and self.trainable_variables support.


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
            OUTPUT = 'output'

    def __init__(self,
                 info,
                 tensors: Dict[str, Tensor] = None,
                 graphs: Dict[str, 'Model'] = None,
                 config: Dict[str, 'Config'] = None):
        super().__init__(
            info,
            tensors=self.make_inputs(tensors),
            graphs=graphs,
            config=config)

    def make_inputs(self, inputs):
        if inputs is None:
            return {}
        if isinstance(inputs, dict):
            return inputs
        if isinstance(inputs, (Tensor, tf.Tensor)):
            return {self.KEYS.TENSOR.INPUT: inputs}
        if isinstance(inputs, (list, tuple)) and all(
                map(lambda t: isinstance(t, (Tensor, tf.Tensor)), inputs)):
            return {self.KEYS.TENSOR.INPUT: inputs}
        raise TypeError("Invalid inputs {}.".format(inputs))

    def complete_inputs(self, inputs):
        for k, v in self.inputs.items():
            if not k in inputs:
                inputs[k] = v
        return inputs

    def cache_inputs(self, inputs):
        if not self._created:
            for k, v in inputs.items():
                if not k in self.inputs:
                    self.inputs[k] = v

    def __call__(self, inputs=None):
        """
        Returns:
            A dict of tensors.
        """
        if not self.is_made:
            self.make(inputs)

        return self.construct(inputs)

    def _make_kernel_with_scope(self, inputs):
        self._created = False
        if inputs == None:
            inputs = {}
        if not isinstance(inputs, Dict):
            inputs = {self.KEYS.TENSOR.INPUT: inputs}
        self.inputs = {}
        self.outputs = {}
        # inputs.update(self.tensors)
        for k, v in self.tensors.items():
            if v is not None and inputs.get(k) is None:
                inputs[k] = v
        self.construct(inputs)
        self._created = True

    def construct(self, inputs):
        is_create = not self._created
        if inputs is None:
            inputs = {}
        inputs = self.pre_kernel(inputs)
        self.cache_inputs(inputs)
        if self.is_short_cut(inputs):
            results = self.outputs
        else:
            with self.info.variable_scope(reuse=not is_create):
                inputs = self.pre_kernel_in_scope(inputs)
                results = self.kernel(inputs)
                results = self.post_kernel_in_scope(results)
            results = self.post_kernel(results)
        self.cache_outputs(results)
        return self.maybe_simpify_outputs(results)

    def is_short_cut(self, inputs):
        if not self._created:
            return False
        without_new_inputs = True
        for k in inputs:
            if not inputs[k] is self.inputs.get(k):
                without_new_inputs = False
                break
        return without_new_inputs

    def kernel(self, inputs):
        return {}

    def pre_kernel(self, inputs):
        inputs = self.make_inputs(inputs)
        inputs = self.complete_inputs(inputs)
        return inputs

    def pre_kernel_in_scope(self, inputs):
        return inputs

    def post_kernel_in_scope(self, results):
        return results

    def maybe_simpify_outputs(self, results):
        if len(results) == 1:
            if self.KEYS.TENSOR.MAIN in results:
                return results[self.KEYS.TENSOR.MAIN]
            if self.KEYS.TENSOR.OUTPUT in results:
                return results[self.KEYS.TENSOR.OUTPUT]
        return results

    def _output_cache_name_map(self, key):
        return key

    def cache_outputs(self, results):
        if not self._created:
            for k, v in results.items():
                self.outputs[k] = v
                if self._output_cache_name_map(k) is not None:
                    self.tensors[self._output_cache_name_map(k)] = v

                # self.tensors['output/{}'.format(k)] = v

    def post_kernel(self, results):
        if not self._created and results is None:
            results = {}
        if results is None:
            return results
        if isinstance(results, (Tensor, tf.Tensor)):
            results = {self.KEYS.TENSOR.MAIN: results}
        return results
