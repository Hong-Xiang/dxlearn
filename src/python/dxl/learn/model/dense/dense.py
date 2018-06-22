import tensorflow as tf
from dxl.learn.core import Model

__all__ = ['Dense']


class Dense(Model):
    class KEYS(Model.KEYS):
        class CONFIG:
            N_UNITS = 'n_units'
            ACTIVATION = 'activation'
            W_INIT = 'w_init'
            B_INIT = 'b_init'

    def __init__(self,
                 info='dense',
                 inputs=None,
                 n_units=None,
                 activation=None,
                 w_init=None,
                 b_init=None):
        KC = self.KEYS.CONFIG
        super().__init__(
            info,
            tensors={self.KEYS.TENSOR.INPUT: inputs},
            config={
                KC.N_UNITS: n_units,
                KC.ACTIVATION: activation,
                KC.W_INIT: w_init,
                KC.B_INIT: b_init
            }
        )
    
    @classmethod
    def _default_config(cls):
        KC = cls.KEYS.CONFIG
        return {
            KC.N_UNITS: 64,
            KC.W_INIT: tf.truncated_normal_initializer(stddev=0.1),
            KC.B_INIT: tf.constant_initializer(value=0.0)
        }

    def kernel(self, inputs):
        KT, KC = self.KEYS.TENSOR, self.KEYS.CONFIG
        x = inputs[KT.INPUT]
        if x.shape.ndims != 2:
            raise AssertionError("The input dimension must be rank 2")
        
        n_in = x.shape.as_list()[-1]

        with tf.variable_scope("init_w_b"):
            w = tf.get_variable(name='w', 
                                shape=(n_in, self.config(KC.N_UNITS)),
                                initializer=self.config(KC.W_INIT),
                                dtype=x.dtype,
                                reuse=self._created)
            y = tf.matmul(x, w)
            if self.config(KC.B_INIT):
                b = tf.get_variable(name='b',
                                    shape=(self.config(KC.N_UNITS),),
                                    initializer=self.config(KC.B_INIT),
                                    dtype=x.dtype,
                                    reuse=self._created)
            y = tf.nn.bias_add(y, b, name='bias_add')

        return y
