from dxl.learn.core import Model, Tensor
import tensorflow as tf
import numpy as np

class DataSplitter(Model):
    """
    split the blockpairs into multiple parts without data loss.
    the last slice may contain more data than others.
    """

    def __init__(self, name, nb_split, graph_info):
        self._nb_split = nb_split
        super().__init__(name, graph_info=graph_info)

    def kernel(self, inputs):
        if len(inputs) == 0:
            return None
        data: tf.Tensor = inputs[self.KEYS.TENSOR.INPUT].data
        data_shape = data.shape.as_list()
        size = data_shape[0] // self._nb_split
        # the last data slice may contain more data.(no data truncation)
        last_size = data_shape[0] - size * (self._nb_split - 1)
        result = {}
        for i in range(self._nb_split - 1):
            result['slice_{}'.format(i)] = tf.slice(data,
                                                    [size * i, 0],
                                                    [size, data_shape[1]])
        # arrange the last slice individully.
        result['slice_{}'.format(self._nb_split - 1)] = tf.slice(data,
                                                                 [size * self._nb_split-1, 0],
                                                                 [last_size, data_shape[1]])
        ginfo = inputs[self.KEYS.TENSOR.INPUT].graph_info
        result = {k: Tensor(result[k], None, ginfo.update(name=ginfo.name + '_{}'.format(k)))
                  for k in result}
        return result


class ProjectionSplitter(Model):
    def __init__(self, name, nb_split, graph_info):
        self._nb_split = nb_split
        super().__init__(name, graph_info=graph_info)

    def kernel(self, inputs):
        if len(inputs) == 0:
            return None
        ip: tf.Tensor = inputs[self.KEYS.TENSOR.INPUT].data
        ip_shape = ip.shape.as_list()
        size = ip_shape[0] // self._nb_split
        result = {}
        for i in range(self._nb_split):
            result['slice_{}'.format(i)] = tf.slice(ip,
                                                    [size * i, 0],
                                                    [size, ip_shape[1]])
        ginfo = inputs[self.KEYS.TENSOR.INPUT].graph_info
        result = {k: Tensor(result[k], None, ginfo.update(name=ginfo.name + '_{}'.format(k)))
                  for k in result}
        return result
