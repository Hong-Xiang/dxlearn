from dxl.learn.core import Model, Tensor
import tensorflow as tf


class ReconStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            PROJECTION = 'projection'
            SYSTEM_MATRIX = 'system_matrix'
            EFFICIENCY_MAP = 'efficiency_map'

    def __init__(self, name, image, projection, system_matrix,
                 efficiency_map, graph_info):
        super().__init__(name,
                         {self.KEYS.TENSOR.IMAGE: image,
                          self.KEYS.TENSOR.PROJECTION: projection,
                          self.KEYS.TENSOR.SYSTEM_MATRIX: system_matrix,
                          self.KEYS.TENSOR.EFFICIENCY_MAP: efficiency_map},
                         graph_info=graph_info)

    def kernel(self, inputs):
        img = inputs[self.KEYS.TENSOR.IMAGE].data
        proj = inputs[self.KEYS.TENSOR.PROJECTION].data
        sm = inputs[self.KEYS.TENSOR.SYSTEM_MATRIX].data
        effmap = inputs[self.KEYS.TENSOR.EFFICIENCY_MAP].data
        px = tf.matmul(sm, img)
        dv = proj / px
        bp = tf.matmul(sm, dv, transpose_a=True)
        result = img / effmap * bp
        return Tensor(result, None, self.graph_info.update(name=None))


class EfficiencyMap(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            SYSTEM_MATRIX: 'system_matrix'

    def __init__(self, name, system_matrix, graph_info):
        super().__init__(name, {self.KEYS.TENSOR.SYSTEM_MATRIX: system_matrix},
                         graph_info=graph_info)

    def kernel(self, inputs):
        sm: Tensor = inputs[self.KEYS.TENSOR.SYSTEM_MATRIX].data
        ones = tf.ones([sm.shape[0], 1])
        return Tensor(tf.matmul(sm, ones, transpose_a=True), None, self.graph_info.update(name=None))


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
