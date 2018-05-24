from dxpy.learn.graph import Graph
from dxpy.configs import configurable
from dxpy.learn.config import config
import tensorflow as tf
import numpy as np

class SinDataset(Graph):
    @configurable(config, with_name=True)
    def __init__(self, name, batch_size=32, amplitude=1.0, frequency=1.0, phase=0.0):
        super().__init__(name=name, batch_size=batch_size, amplitude=amplitude, frequency=frequency, phase=phase)
        self._construct_dataset()

    def _construct_dataset(self):
        from dxpy.learn.utils.tensor import ensure_shape
        with tf.name_scope(str(self.name)):
            x = tf.random_uniform(shape=[self.param('batch_size'), 1], minval=0, maxval=2*np.pi)
            y = tf.sin(x*self.param('frequency')+self.param('phase'))*self.param('amplitude')
            # dataset = tf.data.Dataset.from_tensors({'x': x, 'y': y})
            # dataset = dataset.batch(self.param('batch_size'))
            # iterator = dataset.make_one_shot_iterator()
            # next_element = iterator.get_next()
            shape = [self.param('batch_size'), 1]
            self.register_node('x', ensure_shape(x, shape=shape))
            self.register_node('y', ensure_shape(y, shape=shape))

    

