from dxl.learn.model.base import Model
import tensorflow as tf


class Dense(Model):
    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            HIDDEN = 'hidden'

    def __init__(self, name, hidden):
        super().__init__(name)
        self.config[self.KEYS.CONFIG.HIDDEN] = hidden
        self.model = None

    def build(self, x):
        if isinstance(x, tf.Tensor):
            self.model = tf.layers.Dense(self.config[self.KEYS.CONFIG.HIDDEN])
        else:
            raise TypeError(f"Not support tensor type {type(x)}.")

    def kernel(self, x):
        return self.model(x)

    @property
    def parameters(self):
        return self.model.weights
