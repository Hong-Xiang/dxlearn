from dxl.learn.core import Model, Tensor
import tensorflow as tf
import numpy as np
op = tf.load_op_library(
    '/home/chengaoyu/tools/tensorflow/bazel-bin/tensorflow/core/user_ops/pet_gpu.so')

# mlop = tf.load_op_library(
#     '/home/chengaoyu/tools/tensorflow/bazel-bin/tensorflow/core/user_ops/make_lors.so')

# makelors = mlop.make_lors

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
        gridx = tf.constant(
            np.array([grid[2], grid[0], grid[1]]), name='gridx')
        centerx = tf.constant(
            np.array([center[2], center[0], center[1]]), name='centerx')
        sizex = tf.constant(
            np.array([size[2], size[0], size[1]]), name='sizex')
        px = projection(lors=xlors, image=imgx, grid=gridx,
                        center=centerx, size=sizex, kernel_width=kernel_width, model=model)

        bpx = backprojection(image=imgx, grid=gridx, lors=xlors,
                             center=centerx, size=sizex, line_integral=px,  kernel_width=kernel_width, model=model)
        bpxt = tf.transpose(bpx, perm=[1, 2, 0])

        # y-dominant, tranposed
        # gridy = grid
        # centery = center
        # sizey = size
        gridy = tf.constant(
            np.array([grid[1], grid[0], grid[2]]), name='gridy')
        centery = tf.constant(
            np.array([center[1], center[0], center[2]]), name='centery')
        sizey = tf.constant(
            np.array([size[1], size[0], size[2]]), name='sizey')
        py = projection(lors=ylors, image=imgy, grid=gridy,
                        center=centery, size=sizey, kernel_width=kernel_width, model=model)

        bpy = backprojection(image=imgy, grid=gridy, lors=ylors,
                             center=centery, size=sizey, line_integral=py,  kernel_width=kernel_width, model=model)
        bpyt = tf.transpose(bpy, perm=[1, 0, 2])

        result = imgz / (effmap + 1e-8) * (bpxt + bpyt + bpz)
        # result = imgz / (effmap+1e-8) * bpz
        return Tensor(result, None, self.graph_info.update(name=None))


class Projection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            # PROJECTION = 'projection'
            # SYSTEM_MATRIX = 'system_matrix'
            # EFFICIENCY_MAP = 'efficiency_map'
            GRID = 'grid'
            CENTER = 'center'
            SIZE = 'size'
            LORS_X = 'xlors'
            LORS_Y = 'ylors'
            LORS_Z = 'zlors'

    def __init__(self, name, image, grid, center, size,
                 xlors, ylors, zlors, graph_info):
        super().__init__(name,
                         {self.KEYS.TENSOR.IMAGE: image,
                          self.KEYS.TENSOR.GRID: grid,
                          self.KEYS.TENSOR.CENTER: center,
                          self.KEYS.TENSOR.SIZE: size,
                          self.KEYS.TENSOR.LORS_X: xlors,
                          self.KEYS.TENSOR.LORS_Y: ylors,
                          self.KEYS.TENSOR.LORS_Z: zlors},
                         graph_info=graph_info)

    def kernel(self, inputs):
        imgz = inputs[self.KEYS.TENSOR.IMAGE].data

        imgx = tf.transpose(imgz, perm=[2, 0, 1])
        imgy = tf.transpose(imgz, perm=[1, 0, 2])
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
        pz = projection(lors=zlors, image=imgz, grid=grid,
                        center=center, size=size, kernel_width=kernel_width, model=model)

        gridx = tf.constant(
            np.array([grid[2], grid[0], grid[1]]), name='gridx')
        centerx = tf.constant(
            np.array([center[2], center[0], center[1]]), name='centerx')
        sizex = tf.constant(
            np.array([size[2], size[0], size[1]]), name='sizex')
        px = projection(lors=xlors, image=imgx, grid=gridx,
                        center=centerx, size=sizex, kernel_width=kernel_width, model=model)
        gridy = tf.constant(
            np.array([grid[1], grid[0], grid[2]]), name='gridy')
        centery = tf.constant(
            np.array([center[1], center[0], center[2]]), name='centery')
        sizey = tf.constant(
            np.array([size[1], size[0], size[2]]), name='sizey')
        py = projection(lors=ylors, image=imgy, grid=gridy,
                        center=centery, size=sizey, kernel_width=kernel_width, model=model)
        pxt = Tensor(px, None, self.graph_info.update(name=None))
        pyt = Tensor(py, None, self.graph_info.update(name=None))
        pzt = Tensor(pz, None, self.graph_info.update(name=None))
        return pxt, pyt, pzt


# class MakeLors(Model):
#     """
#     give a block pair and return the valid lors
#     """
#     class KEYS(Model.KEYS):
#         class TENSOR(Model.KEYS.TENSOR):
#             BLOCKPAIR = 'blockpair'

#     def __init__(self, name, blockpair, graph_info):
#         super().__init__(name,
#                          {self.KEYS.TENSOR.BLOCKPAIR: blockpair},
#                          graph_info = graph_info)
#     def kernel(self, inputs):
#         blockpair = inputs[self.KEYS.TENSOR.BLOCKPAIR].data
#         lors = makelors(blockpair)
#         return Tensor(lors, None, self.graph_info.update(name = None))

class BackProjection(Model):
    """
    backproject the projection data on the image along the lors.
    """
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            GRID = 'grid'
            CENTER = 'center'
            SIZE = 'size'
            LORS_X = 'xlors'
            LORS_Y = 'ylors'
            LORS_Z = 'zlors'
            PROJ_X = 'xproj'
            PROJ_Y = 'yproj'
            PROJ_Z = 'zproj'

    def __init__(self, name, image, grid, center, size, xlors, ylors, zlors,
                 xproj, yproj, zproj, graph_info):
        super().__init__(name,
                         {self.KEYS.TENSOR.IMAGE: image,
                          self.KEYS.TENSOR.GRID: grid,
                          self.KEYS.TENSOR.CENTER: center,
                          self.KEYS.TENSOR.SIZE: size,
                          self.KEYS.TENSOR.LORS_X: xlors,
                          self.KEYS.TENSOR.LORS_Y: ylors,
                          self.KEYS.TENSOR.LORS_Z: zlors,
                          self.KEYS.TENSOR.PROJ_X: xproj,
                          self.KEYS.TENSOR.PROJ_Y: yproj,
                          self.KEYS.TENSOR.PROJ_Z: zproj},
                         graph_info=graph_info)

    def kernel(self, inputs):
        imgz = inputs[self.KEYS.TENSOR.IMAGE].data
        imgx = tf.transpose(imgz, perm=[2, 0, 1])
        imgy = tf.transpose(imgz, perm=[1, 0, 2])
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

        xproj = inputs[self.KEYS.TENSOR.PROJ_X].data
        yproj = inputs[self.KEYS.TENSOR.PROJ_Y].data
        zproj = inputs[self.KEYS.TENSOR.PROJ_Z].data

        model = 'tor'
        kernel_width = np.sqrt(3 * 3 / np.pi)

        bpz = backprojection(image=imgz, grid=grid, lors=zlors,
                             center=center, size=size, line_integral=zproj,  kernel_width=kernel_width, model=model)
        
        # grid0 = grid[0]
        # grid1 = grid[1]
        # grid2 = grid[2]
        # gridx = tf.constant(
        #     np.array([grid2, grid0, grid1], dtype = np.int32), name='gridx')
        # centerx = tf.constant(
        #     np.array([center[2], center[0], center[1]], np.float32), name='centerx')
        # sizex = tf.constant(
        #     np.array([size[2], size[0], size[1]], np.float32), name='sizex')

        gridx = inputs[self.KEYS.TENSOR.GRID].data
        centerx = inputs[self.KEYS.TENSOR.CENTER].data
        sizex = inputs[self.KEYS.TENSOR.SIZE].data


        bpx = backprojection(image=imgx, grid=gridx, lors=xlors,
                             center=centerx, size=sizex, line_integral=xproj,  kernel_width=kernel_width, model=model)
        bpxt = tf.transpose(bpx, perm=[1, 2, 0])

        # gridy = tf.constant(
        #     np.array([grid[1], grid[0], grid[2]], np.int32), name='gridy')
        # centery = tf.constant(
        #     np.array([center[1], center[0], center[2]], np.float32), name='centery')
        # sizey = tf.constant(
        #     np.array([size[1], size[0], size[2]], np.float32), name='sizey')
        
        gridy = inputs[self.KEYS.TENSOR.GRID].data
        centery = inputs[self.KEYS.TENSOR.CENTER].data
        sizey = inputs[self.KEYS.TENSOR.SIZE].data

        bpy = backprojection(image=imgy, grid=gridy, lors=ylors,
                             center=centery, size=sizey, line_integral=yproj,  kernel_width=kernel_width, model=model)
        bpyt = tf.transpose(bpy, perm=[1, 0, 2])

        result = (bpxt + bpyt + bpz)
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
