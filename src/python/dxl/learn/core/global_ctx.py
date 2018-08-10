from .graph import Graph
from .tensor import Variable
import numpy as np
import tensorflow as tf
from contextlib import contextmanager
from .session import ThisSession


class GlobalContext(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            KEEP_PROB = 'keep_prob'
            KEEP_PROB_RATIO = 'keep_prob_ratio'

    def __init__(self):
        super().__init__('global_context')

    def kernel(self, inputs=None):
        from dxl.learn.tensor.keep_prob import KeepProb
        self.tensors['keep_prob'] = KeepProb()

    @contextmanager
    def test_phase(self):
        with self.tensors['keep_prob'].test_phase():
            yield


global_context = GlobalContext()


def get_global_context():
    return global_context


class GlobalContextV2(GlobalContext):
    ...