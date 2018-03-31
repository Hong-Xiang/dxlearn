from dxl.learn.core import Model, Tensor
# from dxl.learn.model.tor_recon import Projection, BackProjection
import tensorflow as tf
import numpy as np
import warnings

warnings.warn(DeprecationWarning())

projection = op.projection_gpu
backprojection = op.backprojection_gpu

# class ReconStep1(Model):
#   class KEYS(Model.KEYS):
#     class TENSOR(Model.KEYS.TENSOR):
#       IMAGE = 'image'
#       EFFICIENCY_MAP = 'efficiency_map'
#       LORS_X = 'xlors'
#       LORS_Y = 'ylors'
#       LORS_Z = 'zlors'

#   def __init__(self, name, image, efficiency_map, grid, center, size, xlors,
#                ylors, zlors, graph_info):
#     self.grid = grid
#     self.center = center
#     self.size = size
#     self.eps = 1e-8
#     super().__init__(
#         name, {
#             self.KEYS.TENSOR.IMAGE: image,
#             self.KEYS.TENSOR.EFFICIENCY_MAP: efficiency_map,
#             self.KEYS.TENSOR.LORS_X: xlors,
#             self.KEYS.TENSOR.LORS_Y: ylors,
#             self.KEYS.TENSOR.LORS_Z: zlors
#         },
#         graph_info=graph_info)

#   def kernel(self, inputs):
#     img = self.tensor(self.KEYS.TENSOR.IMAGE)
#     projections = Projection(
#         'projection',
#         img,
#         self.grid,
#         self.center,
#         self.size,
#         self.tensor(self.KEYS.TENSOR.LORS_X),
#         self.tensor(self.KEYS.TENSOR.LORS_Y),
#         self.tensor(self.KEYS.TENSOR.LORS_Z),
#         self.graph_info.update(name=None))()
#     backprojections = BackProjection(
#         'backprojection',
#         img,
#         self.grid,
#         self.center,
#         self.size,
#         self.tensor(self.KEYS.TENSOR.LORS_X),
#         self.tensor(self.KEYS.TENSOR.LORS_Y),
#         self.tensor(self.KEYS.TENSOR.LORS_Z),
#         projections['x'],
#         projections['y'],
#         projections['z'],
#         self.graph_info.update(name=None))()
#     new_img = sum(map(lambda t: t.data, backprojections.values()))
#     emap = self.tensor(self.KEYS.TENSOR.EFFICIENCY_MAP).data
#     result = img.data / (emap + 1e-8) * new_img
#     return Tensor(result, None, self.graph_info.update(name=None))


class ReconStep(Model):
  class KEYS(Model.KEYS):
    class TENSOR(Model.KEYS.TENSOR):
      IMAGE = 'image'
      # PROJECTION = 'projection'
      # SYSTEM_MATRIX = 'system_matrix'
      EFFICIENCY_MAP = 'efficiency_map'
      LORS_X = 'xlors'
      LORS_Y = 'ylors'
      LORS_Z = 'zlors'

  def __init__(self, name, image, efficiency_map, grid, center, size, xlors,
               ylors, zlors, graph_info):
    self.grid = np.array(grid, dtype=np.int32)
    self.center = np.array(center, dtype=np.float32)
    self.size = np.array(size, dtype=np.float32)
    super().__init__(
        name,
        {
            self.KEYS.TENSOR.IMAGE:
            image,
            #   self.KEYS.TENSOR.PROJECTION: projection,
            #   self.KEYS.TENSOR.SYSTEM_MATRIX: system_matrix,
            self.KEYS.TENSOR.EFFICIENCY_MAP:
            efficiency_map,
            self.KEYS.TENSOR.LORS_X:
            xlors,
            self.KEYS.TENSOR.LORS_Y:
            ylors,
            self.KEYS.TENSOR.LORS_Z:
            zlors
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

    # grid = inputs[self.KEYS.TENSOR.GRID].data
    # center = inputs[self.KEYS.TENSOR.CENTER].data
    # size = inputs[self.KEYS.TENSOR.SIZE].data

    grid = self.grid
    center = self.center
    size = self.size
    xlors = inputs[self.KEYS.TENSOR.LORS_X].data
    ylors = inputs[self.KEYS.TENSOR.LORS_Y].data
    zlors = inputs[self.KEYS.TENSOR.LORS_Z].data

    # lors tranposed
    xlors = tf.transpose(xlors)
    ylors = tf.transpose(ylors)
    zlors = tf.transpose(zlors)

    model = 'tor'
    crystal_size = size[2] / grid[2]
    kernel_width = np.sqrt(crystal_size * crystal_size / np.pi)
    # px = tf.matmul(sm, img)
    # replaced with tor projection

    # z-dominant, no transpose
    pz = projection(
        lors=zlors,
        image=imgz,
        grid=grid,
        center=center,
        size=size,
        kernel_width=kernel_width,
        model=model)

    bpz = backprojection(
        image=imgz,
        grid=grid,
        lors=zlors,
        center=center,
        size=size,
        line_integral=pz,
        kernel_width=kernel_width,
        model=model)
    # x-dominant, tranposed
    gridx = tf.constant(np.array([grid[0], grid[2], grid[1]]), name='gridx')
    centerx = tf.constant(
        np.array([center[0], center[2], center[1]]), name='centerx')
    sizex = tf.constant(np.array([size[0], size[2], size[1]]), name='sizex')
    px = projection(
        lors=xlors,
        image=imgx,
        grid=gridx,
        center=centerx,
        size=sizex,
        kernel_width=kernel_width,
        model=model)

    bpx = backprojection(
        image=imgx,
        grid=gridx,
        lors=xlors,
        center=centerx,
        size=sizex,
        line_integral=px,
        kernel_width=kernel_width,
        model=model)
    bpxt = tf.transpose(bpx, perm=[1, 2, 0])

    # y-dominant, tranposed
    # gridy = grid
    # centery = center
    # sizey = size
    gridy = tf.constant(np.array([grid[1], grid[2], grid[0]]), name='gridy')
    centery = tf.constant(
        np.array([center[1], center[2], center[0]]), name='centery')
    sizey = tf.constant(np.array([size[1], size[2], size[0]]), name='sizey')
    py = projection(
        lors=ylors,
        image=imgy,
        grid=gridy,
        center=centery,
        size=sizey,
        kernel_width=kernel_width,
        model=model)

    bpy = backprojection(
        image=imgy,
        grid=gridy,
        lors=ylors,
        center=centery,
        size=sizey,
        line_integral=py,
        kernel_width=kernel_width,
        model=model)
    bpyt = tf.transpose(bpy, perm=[1, 0, 2])

    result = imgz / (effmap + 1e-8) * (bpxt + bpyt + bpz)
    # result = imgz / (effmap+1e-8) * bpz
    return Tensor(result, None, self.graph_info.update(name=None))
