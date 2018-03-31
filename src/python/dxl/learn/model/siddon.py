from dxl.learn.core import Model, Tensor
import tensorflow as tf
import numpy as np
op = tf.load_op_library(
    '/home/chengaoyu/tools/tensorflow/bazel-bin/tensorflow/core/user_ops/siddon_gpu.so')

# mlop = tf.load_op_library(
#     '/home/chengaoyu/tools/tensorflow/bazel-bin/tensorflow/core/user_ops/make_lors.so')

# makelors = mlop.make_lors

projection = op.projection_gpu
backprojection = op.backprojection_gpu


class Projection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            LORS = 'lors'

    def __init__(self, name, image,
                 grid, center, size,
                 lors, tof_bin, time_resolution,
                 graph_info):
        self.grid = grid
        self.origin = origin
        self.size = size
        super().__init__(
            name, {
                self.KEYS.TENSOR.IMAGE: image,
                self.KEYS.TENSOR.LORS:  lors
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        grid = self.grid
        center = self.center
        size = self.size
        image = inputs[self.KEYS.TENSOR.IMAGE].data
        lors = inputs[self.KEYS.TENSOR.LORS].data
        model = 'siddon'

        def projection_axis():
            return projection(
                lors=lors,
                image=image,
                grid=grid,
                origin=origin,
                size=size,
                #   kernel_width=kernel_width,
                model=model)

        projections = projections_axis()
        return Tensor(projections, None, self.graph_info.update(name=None))


class BackProjection(Model):
    """
      backproject the projection data on the image along the lors.
      """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            LORS = 'lors'
            PROJ = 'proj'

    def __init__(self, name, image,
                 grid, origin, size,
                 tof_bin, time_resolution,
                 lors, proj, graph_info):
        self.grid = grid
        self.origin = origin
        self.size = size
        super().__init__(
            name, {
                self.KEYS.TENSOR.IMAGE: image,
                self.KEYS.TENSOR.LORS: lors,
                self.KEYS.TENSOR.PROJ: proj
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        image = inputs[self.KEYS.TENSOR.IMAGE].data
        lors = inputs[self.KEYS.TENSOR.LORS].data
        projections = inputs[self.KEYS.TENSOR.PROJ].data

        model = 'siddon'
        # kernel_width = KERNEL_WIDTH

        def backprojection_axis():
            return backprojection(
                image=image,
                grid=self.grid,
                lors=lors,
                origin=self.origin,
                size=self.size,
                lor_values=projections,
                tof_bin=1e-15,
                time_resolution=20000,
                model=model)

        backprojection = backprojection_axis()
        return Tensor(backprojection, None, self.graph_info.update(name=None))


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
