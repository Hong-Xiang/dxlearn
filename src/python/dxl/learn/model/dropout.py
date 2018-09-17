from dxl.learn.core.global_ctx import get_global_context
import tensorflow as tf
from dxl.learn.model.base import Model


class DropOut(Model):
    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            KEEP_PROB = 'keep_prob'

    def __init__(self, keep_prob=None, name='DEFAULT_DROPOUT'):
        """
        `keep_prob`: keep probability during training.
        """
        super().__init__(name)
        self.prob = keep_prob

    @property
    def parameters(self):
        return []

    def kernel(self, x):
        return tf.nn.dropout(x, self.prob)

    def build(self, x):
        pass
