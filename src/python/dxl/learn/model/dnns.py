from .base import Model
import tensorflow as tf


class Dense(Model):
    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            HIDDEN = 'hidden'

    def __init__(self, hidden, name='dense'):
        super().__init__(name)
        self._config[self.KEYS.CONFIG.HIDDEN] = hidden
        self.model = None

    def build(self, x):
        if isinstance(x, tf.Tensor):
            self.model = tf.layers.Dense(self.config[self.KEYS.CONFIG.HIDDEN])
        else:
            raise TypeError(f"Not support tensor type {type(x)}.")

    def kernel(self, x):
        return self.model(x)
