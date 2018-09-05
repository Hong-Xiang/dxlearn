from dxl.learn.model.base import Model
import tensorflow as tf


class Dense(Model):
    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            HIDDEN = 'hidden'

    def __init__(self, hidden, name='dense'):
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

if __name__ == '__main__':
    x = tf.ones([32, 2],dtype=tf.float32)
    a = Dense(128)
    #a = tf.layers.Dense(128)
    y = a(x)
    print(a.parameters)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y1 = sess.run(y)
        print(y1)