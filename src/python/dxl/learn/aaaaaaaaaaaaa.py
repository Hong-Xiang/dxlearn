import numpy as np
from doufo import List
from pathlib import Path
import os
import platform
import tensorflow as tf
from dxl.learn.model import Conv2D,Dense,Inception,Model


class Merge(Model):
    def __init__(self, filters, name='merger'):
        super().__init__(name)
        self.filters = filters
        self.model = Conv2D(filters, 3,name='conv_merge')

    @property
    def parameters(self):
        return self.model.parameters

    def kernel(self, x):
        x = tf.concat(x,axis=3)
        return self.model(x)

    def build(self,x):
        pass

if __name__ == '__main__':
    x = tf.constant(np.ones([32, 64, 64, 3], np.float32))
    m1 = Conv2D(64, 3, name='conv')
    m2 = Merge(32)
    #m2 = Dense(128)
    paths = []
    for i in range(3):
        paths.append(Conv2D(32 + 32 * i, 3, name='conv' + str(i)))

    m = Inception('inception1', m1, paths, m2)
    y = m(x)
    print(m.parameters)
    print(y)