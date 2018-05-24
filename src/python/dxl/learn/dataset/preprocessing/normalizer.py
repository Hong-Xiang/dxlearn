import tensorflow as tf
import numpy as np
from ...graph import Graph, NodeKeys


class Normalizer(Graph):

    def __init__(self, name, input_tensors=None, **config):
        super().__init__(name, **config)
        self.register_node(NodeKeys.INPUT, input_tensor)

        self.register_main_task(self.__normalization)
        self.register_task('denorm', self.__denormalization)
        self.register_task('norm', self.__normalization)

    def __unified_input_tensors(self, input_tensors):
        pass

    def _normalization_kernel(self, feeds):
        raise NotImplementedError

    def _denormalization_kernel(self, feeds):
        raise NotImplementedError

    def __normalization(self, feeds):
        return self._normalization_kernel(self.get_feed_dict(feeds))

    def __denormalization(self, feeds):
        return self._denormalization_kernel(self.get_feed_dict(feeds))


class FixWhite(Normalizer):
    method_name = 'fix_white'

    def __init__(self, name, input_tensor, **config):
        super().__init__(name, input_tensor, **config)

    def _normalization_kernel(self, feeds):
        with tf.name_scope('normalization'):
            return (feeds - self.c['mean']) / self.c['std']

    def _denormalization_kernel(self, feeds):
        with tf.name_scope('denormalization'):
            return feeds * self.c['std'] + self.c['mean']


class MeanStdWhite(Normalizer):
    method_name = 'mean_std_white'

    def __init__(self, name, input_tensor, **config):
        super().__init__(name, input_tensor, **config)

    def _normalization_kernel(self, feeds):
        with tf.name_scope('normalization'):
            mean = tf.reduce_mean(feeds)
            maxb = tf.reduce_max(feeds)
            minb = tf.reduce_min(feeds)
            return (feeds - self.c['mean']) / self.c['std']


class SelfMinMax(Normalizer):
    method_name = 'self_min_max'

    def __init__(self, name, input_tensor, **config):
        super().__init__(name, input_tensor, **config)

    def _normalization_kernel(self, feeds):
        if isinstance(feeds, tf.Tensor):
            with tf.name_scope('normalization'):
                rmin = tf.reduce_min(feeds)
                rmax = tf.reduce_max(feeds)
                data = (feeds - rmin) / (rmax - rmin)
        else:
            rmin = np.min(feeds)
            rmax = np.max(feeds)
            data = (feeds - rmin) / (rmax - rmin)
        return {'min': rmin, 'max': rmax, 'data': data}

    def _denormalization_kernel(self, feeds):
        with tf.name_scope('denormalization'):
            return feeds['data'] * (feeds['max'] - feeds['min']) + feeds['min']


class SelfMeanStd(Normalizer):
    method_name = 'self_mean_std'

    def __init__(self, name, input_tensor, **config):
        super().__init__(name, input_tensor, **config)

    def _normalization_kernel(self, feeds):
        if isinstance(feeds, tf.Tensor):
            with tf.name_scope('normalization'):
                mean, stdv = tf.nn.moments(feeds)
                data = (feeds - mean) / stdv
        else:
            mean = np.mean(feeds)
            stdv = np.std(feeds)
            data = (feeds - mean) / stdv
        return {'mean': mean, 'std': stdv, 'data': data}

    def _denormalization_kernel(self, feeds):
        with tf.name_scope('denormalization'):
            return feeds['data'] * feeds['std'] + feeds['mean']


class ReduceSum(Normalizer):
    def __init__(self, name='normalizer/reduce_sum', **config):
        super().__init__(name, **config)
