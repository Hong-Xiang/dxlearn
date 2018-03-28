from dxl.learn.core import Model, Tensor
import tensorflow as tf
import numpy as np
op = tf.load_op_library(
    '/home/hongxwing/Downloads/tensorflow/bazel-bin/tensorflow/core/user_ops/pet_gpu.so')

projection = op.projection_gpu
backprojection = op.backprojection_gpu


class ReconStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            # PROJECTION = 'projection'
            # SYSTEM_MATRIX = 'system_matrix'
            EFFICIENCY_MAP = 'efficiency_map'
            GRID = 'grid'
            CENTER = 'center'
            SIZE = 'size'
            LORS_X = 'xlors'
            LORS_Y = 'ylors'
            LORS_Z = 'zlors'

    def __init__(self, name, image, efficiency_map,
                 grid, center, size,
                 xlors, ylors, zlors, graph_info):
        super().__init__(name,
                         {self.KEYS.TENSOR.IMAGE: image,
                          #   self.KEYS.TENSOR.PROJECTION: projection,
                          #   self.KEYS.TENSOR.SYSTEM_MATRIX: system_matrix,
                          self.KEYS.TENSOR.EFFICIENCY_MAP: efficiency_map,
                          self.KEYS.TENSOR.GRID: grid,
                          self.KEYS.TENSOR.CENTER: center,
                          self.KEYS.TENSOR.SIZE: size,
                          self.KEYS.TENSOR.LORS_X: xlors,
                          self.KEYS.TENSOR.LORS_Y: ylors,
                          self.KEYS.TENSOR.LORS_Z: zlors
                          },
                         graph_info=graph_info)

    def kernel(self, inputs):
        # the default order of the image is z-dominant(z,y,x)
        # for projection another two images are created.
        imgz = inputs[self.KEYS.TENSOR.IMAGE].data

        imgx = tf.transpose(imgz, perm=[2, 0, 1])
        imgy = tf.transpose(imgz, perm=[1, 0, 2])

        # proj = inputs[self.KEYS.TENSOR.PROJECTION].data
        # sm = inputs[self.KEYS.TENSOR.SYSTEM_MATRIX].data

        effmap = inputs[self.KEYS.TENSOR.EFFICIENCY_MAP].data
        grid = inputs[self.KEYS.TENSOR.GRID].data
        center = inputs[self.KEYS.TENSOR.CENTER].data
        size = inputs[self.KEYS.TENSOR.SIZE].data
        xlors = inputs[self.KEYS.TENSOR.LORS_X].data
        ylors = inputs[self.KEYS.TENSOR.LORS_Y].data
        zlors = inputs[self.KEYS.TENSOR.LORS_Z].data

        # lors tranposed
        xlors = tf.transpose(xlors)
        ylors = tf.transpose(ylors)
        zlors = tf.transpose(zlors)

        model = 'tor'
        kernel_width = np.sqrt(3 * 3 / np.pi)
        # px = tf.matmul(sm, img)
        # replaced with tor projection

        # z-dominant, no transpose
        pz = projection(lors=zlors, image=imgz, grid=grid,
                        center=center, size=size, kernel_width=kernel_width, model=model)

        bpz = backprojection(image=imgz, grid=grid, lors=zlors,
                             center=center, size=size, line_integral=pz,  kernel_width=kernel_width, model=model)
        # x-dominant, tranposed
        gridx = grid
        centerx = center
        sizex = size
        px = projection(lors=xlors, image=imgx, grid=gridx,
                        center=centerx, size=sizex, kernel_width=kernel_width, model=model)

        bpx = backprojection(image=imgx, grid=gridx, lors=xlors,
                             center=centerx, size=sizex, line_integral=px,  kernel_width=kernel_width, model=model)
        bpxt = tf.transpose(bpx, perm=[1, 2, 0])

        # y-dominant, tranposed
        gridy = grid
        centery = center
        sizey = size
        py = projection(lors=ylors, image=imgy, grid=gridy,
                        center=centery, size=sizey, kernel_width=kernel_width, model=model)

        bpy = backprojection(image=imgy, grid=gridy, lors=ylors,
                             center=centery, size=sizey, line_integral=py,  kernel_width=kernel_width, model=model)
        bpyt = tf.transpose(bpy, perm=[1, 0, 2])

        result = imgz / (effmap + 1e-8) * (bpxt + bpyt + bpz)
        # result = imgz / (effmap+1e-8) * bpz
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
