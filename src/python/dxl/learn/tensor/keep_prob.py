from dxl.learn.core import Variable, ThisSession
import tensorflow as tf

from contextlib import contextmanager


class KeepProb(Variable):
    def __init__(self, info='keep_prob', value=0.5):
        super().__init__(info, [], tf.float32, initializer=value)
        self.assign_to_one = self.assign(1.0)
        self.assign_to_init = self.assign(value)

    @contextmanager
    def test_phase(self):
        ThisSession.run(self.assign_to_one)
        yield
        ThisSession.run(self.assign_to_init)
