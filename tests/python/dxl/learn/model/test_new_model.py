import tensorflow as tf
import scipy
import numpy as np
from dxl.learn.model.cnn.blocksv2 import *
from doufo import identity
from dxl.learn.model.base import Stack, as_model, Model
from doufo import List
from doufo.tensor import shape


def test_new_model():
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    m = Conv2D('conv', 32, 3)
    y = m(x)

    def incept_path(ipath, filters):
        result = List([])
        result.append(as_model(tf.nn.elu))
        result.append(Conv2D(f'conv_in_{ipath}', filters, 1))
        for i in range(ipath):
            result.append(as_model(tf.nn.elu))
            result.append(Conv2D(f"conv_{ipath}_{i}", filters, 3))
        return Stack(result)

    class Merge(Model):
        def __init__(self, filters, name='merger'):
            super().__init__(name)
            self.filters = filters
            self.model = Conv2D('conv_merge', filters, 3)

        @property
        def parameters(self):
            return self.model.parameters

        def kernel(self, xs):
            x = tf.concat(xs, axis=3)
            return self.model(x)

    r = Residual('residual',
                 Inception('incept', identity,
                           [incept_path(i, 32) for i in range(3)],
                           Merge(32)),
                 ratio=0.3)
    z = r(y)
    assert shape(z) == [32, 64, 64, 32]
