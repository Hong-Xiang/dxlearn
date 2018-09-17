from .learn_extend import variable_from_tensor, assign
from dxl.learn.session import ThisSession
import tensorflow as tf
from contextlib import contextmanager
import numpy as np


class KeepProb:
    def __init__(self, name='keep_prob', value=0.5, backend=None):
        self.data = variable_from_tensor[backend](np.float32(value))
        self.assign_to_one = assign(self.data, 1.0)
        self.assign_to_init = assign(self.data, value)

    @contextmanager
    def test_phase(self):
        ThisSession.run(self.assign_to_one)
        yield
        ThisSession.run(self.assign_to_init)

    def set_prob(self, prob):
        self.data = prob
