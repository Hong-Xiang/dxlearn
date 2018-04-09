from dxl.learn.core import Model, Tensor
import tensorflow as tf
import numpy as np
from enum import Enum


import warnings

warnings.warn(DeprecationWarning())
import os
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')
op = tf.load_op_library(
    TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/pet_gpu.so')

projection = op.projection_gpu
backprojection = op.backprojection_gpu

ALL_AXIS = ['x', 'y', 'z']

ROTATIONS = {'x': [1, 2, 0], 'y': [0, 2, 1], 'z': [0, 1, 2]}
BACK_ROTATIONS = {'x': [2, 0, 1], 'y': [0, 2, 1], 'z': [0, 1, 2]}
ROTATIONS_IMAGE = {'x': [2, 0, 1], 'y': [1, 0, 2], 'z': [0, 1, 2]}
BACK_ROTATIONS_IMAGE = {'x': [1, 2, 0], 'y': [1, 0, 2], 'z': [0, 1, 2]}
KERNEL_WIDTH = np.sqrt(3.0 * 3.0 * np.pi)


def rotate(x, axis: str):
  if axis.lower() in ROTATIONS:
    return [x[i] for i in ROTATIONS[axis.lower()]]
  raise ValueError("Unsupported axis: {}.".format(axis))


def indices(a, offset=0):
  return [i + offset for i in ROTATIONS[a]]


# class ReconStep(Model):
#   class KEYS(Model.KEYS):
#     class TENSOR(Model.KEYS.TENSOR):
#       IMAGE = 'image'
#       # PROJECTION = 'projection'
#       # SYSTEM_MATRIX = 'system_matrix'
#       EFFICIENCY_MAP = 'efficiency_map'
#       GRID = 'grid'
#       CENTER = 'center'
#       SIZE = 'size'
#       LORS_X = 'xlors'
#       LORS_Y = 'ylors'
#       LORS_Z = 'zlors'

#   def __init__(self, name, image, efficiency_map, grid, center, size, xlors,
#                ylors, zlors, graph_info):
#     super().__init__(
#         name,
#         {
#             self.KEYS.TENSOR.IMAGE:
#             image,
#             #   self.KEYS.TENSOR.PROJECTION: projection,
#             #   self.KEYS.TENSOR.SYSTEM_MATRIX: system_matrix,
#             self.KEYS.TENSOR.EFFICIENCY_MAP:
#             efficiency_map,
#             self.KEYS.TENSOR.GRID:
#             grid,
#             self.KEYS.TENSOR.CENTER:
#             center,
#             self.KEYS.TENSOR.SIZE:
#             size,
#             self.KEYS.TENSOR.LORS_X:
#             xlors,
#             self.KEYS.TENSOR.LORS_Y:
#             ylors,
#             self.KEYS.TENSOR.LORS_Z:
#             zlors
#         },
#         graph_info=graph_info)

#   def kernel(self, inputs):
#     # the default order of the image is z-dominant(z,y,x)
#     # for projection another two images are created.
#     imgz = inputs[self.KEYS.TENSOR.IMAGE].data
#     imgx = tf.transpose(imgz, perm=[2, 0, 1])
#     imgy = tf.transpose(imgz, perm=[1, 0, 2])

#     # proj = inputs[self.KEYS.TENSOR.PROJECTION].data
#     # sm = inputs[self.KEYS.TENSOR.SYSTEM_MATRIX].data

#     effmap = inputs[self.KEYS.TENSOR.EFFICIENCY_MAP].data
#     grid = inputs[self.KEYS.TENSOR.GRID].data
#     center = inputs[self.KEYS.TENSOR.CENTER].data
#     size = inputs[self.KEYS.TENSOR.SIZE].data
#     xlors = inputs[self.KEYS.TENSOR.LORS_X].data
#     ylors = inputs[self.KEYS.TENSOR.LORS_Y].data
#     zlors = inputs[self.KEYS.TENSOR.LORS_Z].data

#     # lors tranposed
#     xlors = tf.transpose(xlors)
#     ylors = tf.transpose(ylors)
#     zlors = tf.transpose(zlors)

#     model = 'tor'
#     kernel_width = np.sqrt(3 * 3 / np.pi)
#     # px = tf.matmul(sm, img)
#     # replaced with tor projection

#     # z-dominant, no transpose
#     pz = projection(
#         lors=zlors,
#         image=imgz,
#         grid=grid,
#         center=center,
#         size=size,
#         kernel_width=kernel_width,
#         model=model)

#     bpz = backprojection(
#         image=imgz,
#         grid=grid,
#         lors=zlors,
#         center=center,
#         size=size,
#         line_integral=pz,
#         kernel_width=kernel_width,
#         model=model)
#     # x-dominant, tranposed
#     gridx = tf.constant(np.array([grid[2], grid[0], grid[1]]), name='gridx')
#     centerx = tf.constant(
#         np.array([center[2], center[0], center[1]]), name='centerx')
#     sizex = tf.constant(np.array([size[2], size[0], size[1]]), name='sizex')
#     px = projection(
#         lors=xlors,
#         image=imgx,
#         grid=gridx,
#         center=centerx,
#         size=sizex,
#         kernel_width=kernel_width,
#         model=model)

#     bpx = backprojection(
#         image=imgx,
#         grid=gridx,
#         lors=xlors,
#         center=centerx,
#         size=sizex,
#         line_integral=px,
#         kernel_width=kernel_width,
#         model=model)
#     bpxt = tf.transpose(bpx, perm=[1, 2, 0])

#     # y-dominant, tranposed
#     # gridy = grid
#     # centery = center
#     # sizey = size
#     gridy = tf.constant(np.array([grid[1], grid[0], grid[2]]), name='gridy')
#     centery = tf.constant(
#         np.array([center[1], center[0], center[2]]), name='centery')
#     sizey = tf.constant(np.array([size[1], size[0], size[2]]), name='sizey')
#     py = projection(
#         lors=ylors,
#         image=imgy,
#         grid=gridy,
#         center=centery,
#         size=sizey,
#         kernel_width=kernel_width,
#         model=model)

#     bpy = backprojection(
#         image=imgy,
#         grid=gridy,
#         lors=ylors,
#         center=centery,
#         size=sizey,
#         line_integral=py,
#         kernel_width=kernel_width,
#         model=model)
#     bpyt = tf.transpose(bpy, perm=[1, 0, 2])

#     result = imgz / (effmap + 1e-8) * (bpxt + bpyt + bpz)
#     # result = imgz / (effmap+1e-8) * bpz
#     return Tensor(result, None, self.graph_info.update(name=None))


class Projection(Model):
  class KEYS(Model.KEYS):
    class TENSOR(Model.KEYS.TENSOR):
      IMAGE = 'image'
      LORS_X = 'xlors'
      LORS_Y = 'ylors'
      LORS_Z = 'zlors'

  def __init__(self, name, image, grid, center, size, xlors, ylors, zlors,
               graph_info):
    self.grid = grid
    self.center = center
    self.size = size
    super().__init__(
        name, {
            self.KEYS.TENSOR.IMAGE: image,
            self.KEYS.TENSOR.LORS_X: xlors,
            self.KEYS.TENSOR.LORS_Y: ylors,
            self.KEYS.TENSOR.LORS_Z: zlors
        },
        graph_info=graph_info)

  def kernel(self, inputs):
    grid = self.grid
    center = self.center
    size = self.size
    imgz = tf.transpose(inputs[self.KEYS.TENSOR.IMAGE].data)
    imgx = tf.transpose(imgz, perm=ROTATIONS_IMAGE['x'])
    imgy = tf.transpose(imgz, perm=ROTATIONS_IMAGE['y'])
    imgs = {'x': imgx, 'y': imgy, 'z': imgz}
    xlors = inputs[self.KEYS.TENSOR.LORS_X].data
    ylors = inputs[self.KEYS.TENSOR.LORS_Y].data
    zlors = inputs[self.KEYS.TENSOR.LORS_Z].data

    # lors tranposed

    lors = {'x': xlors, 'y': ylors, 'z': zlors}

    for a in ['x', 'y']:
      # rotate axis
      slices_inds = indices(a) + indices(a, 3)
      slices = [lors[a][:, i] for i in slices_inds]
      lors[a] = tf.stack(slices)
    lors['z'] = tf.transpose(lors['z'][:, :6])
    # lors = {k: tf.transpose(lors[k]) for k in lors}

    # xlors = xlors[:, [1, 2, 0, 4, 5, 3]]
    # ylors = ylors[:, [0, 2, 1, 3, 5, 4]]

    model = 'tor'
    kernel_width = KERNEL_WIDTH

    def projection_axis(axis):
      return projection(
          lors=lors[axis],
          image=imgs[axis],
          grid=rotate(grid, axis),
          center=rotate(center, axis),
          size=rotate(size, axis),
          kernel_width=kernel_width,
          model=model)

    projections = {k: projection_axis(k) for k in ALL_AXIS}

    return {
        a: Tensor(projections[a], None, self.graph_info.update(name=None))
        for a in ALL_AXIS
    }


class BackProjection(Model):
  """
    backproject the projection data on the image along the lors.
    """

  class KEYS(Model.KEYS):
    class TENSOR(Model.KEYS.TENSOR):
      IMAGE = 'image'
      LORS_X = 'xlors'
      LORS_Y = 'ylors'
      LORS_Z = 'zlors'
      PROJ_X = 'xproj'
      PROJ_Y = 'yproj'
      PROJ_Z = 'zproj'

  def __init__(self, name, image, grid, center, size, xlors, ylors, zlors,
               xproj, yproj, zproj, graph_info):
    self.grid = grid
    self.center = center
    self.size = size
    super().__init__(
        name, {
            self.KEYS.TENSOR.IMAGE: image,
            self.KEYS.TENSOR.LORS_X: xlors,
            self.KEYS.TENSOR.LORS_Y: ylors,
            self.KEYS.TENSOR.LORS_Z: zlors,
            self.KEYS.TENSOR.PROJ_X: xproj,
            self.KEYS.TENSOR.PROJ_Y: yproj,
            self.KEYS.TENSOR.PROJ_Z: zproj
        },
        graph_info=graph_info)

  def kernel(self, inputs):
    imgz = tf.transpose(inputs[self.KEYS.TENSOR.IMAGE].data)
    imgs = {
        'x': tf.transpose(imgz, perm=ROTATIONS_IMAGE['x']),
        'y': tf.transpose(imgz, perm=ROTATIONS_IMAGE['y']),
        'z': imgz
    }

    xlors = inputs[self.KEYS.TENSOR.LORS_X].data
    ylors = inputs[self.KEYS.TENSOR.LORS_Y].data
    zlors = inputs[self.KEYS.TENSOR.LORS_Z].data
    lors = {'x': xlors, 'y': ylors, 'z': zlors}
    for a in ['x', 'y']:
      # rotate axis
      slices_inds = indices(a) + indices(a, 3)
      slices = [lors[a][:, i] for i in slices_inds]
      lors[a] = tf.stack(slices)
    # lors = {k: tf.transpose(lors[k]) for k in lors}
    lors['z'] = tf.transpose(lors['z'][:, :6])

    projections = {
        'x': inputs[self.KEYS.TENSOR.PROJ_X].data,
        'y': inputs[self.KEYS.TENSOR.PROJ_Y].data,
        'z': inputs[self.KEYS.TENSOR.PROJ_Z].data
    }

    model = 'tor'
    kernel_width = KERNEL_WIDTH

    def backprojection_axis(a):
      return backprojection(
          image=imgs[a],
          grid=rotate(self.grid, a),
          lors=lors[a],
          center=rotate(self.center, a),
          size=rotate(self.size, a),
          line_integral=projections[a],
          kernel_width=kernel_width,
          model=model)

    backprojections = {a: backprojection_axis(a) for a in ALL_AXIS}
    backprojections = {
        a: tf.transpose(backprojections[a], BACK_ROTATIONS_IMAGE[a])
        for a in ALL_AXIS
    }

    return {
        a: Tensor(tf.transpose(v), None, self.graph_info.update(name=None))
        for a, v in backprojections.items()
    }


class EfficiencyMap(Model):
  class KEYS(Model.KEYS):
    class TENSOR(Model.KEYS.TENSOR):
      LORS_X = 'xlors'
      LORS_Y = 'ylors'
      LORS_Z = 'zlors'
  

  def __init__(self, name, xlors, ylors, zlors, grid, center, size, graph_info):
    self.grid = grid
    self.center = center
    self.size = size
    super().__init__(
        name, {
          self.KEYS.TENSOR.LORS_X: xlors,
          self.KEYS.TENSOR.LORS_Y: ylors,
          self.KEYS.TENSOR.LORS_Z: zlors,
        },
        graph_info=graph_info)

  def kernel(self, inputs):

    sm: Tensor = inputs[self.KEYS.TENSOR.SYSTEM_MATRIX].data
    ones = tf.ones([sm.shape[0], 1])
    return Tensor(
        tf.matmul(sm, ones, transpose_a=True),
        None,
        self.graph_info.update(name=None))


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
      result['slice_{}'.format(i)] = tf.slice(data, [size * i, 0],
                                              [size, data_shape[1]])
    # arrange the last slice individully.
    result['slice_{}'.format(self._nb_split - 1)] = tf.slice(
        data, [size * self._nb_split - 1, 0], [last_size, data_shape[1]])
    ginfo = inputs[self.KEYS.TENSOR.INPUT].graph_info
    result = {
        k: Tensor(
            result[k], None, ginfo.update(name=ginfo.name + '_{}'.format(k)))
        for k in result
    }
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
      result['slice_{}'.format(i)] = tf.slice(ip, [size * i, 0],
                                              [size, ip_shape[1]])
    ginfo = inputs[self.KEYS.TENSOR.INPUT].graph_info
    result = {
        k: Tensor(
            result[k], None, ginfo.update(name=ginfo.name + '_{}'.format(k)))
        for k in result
    }
    return result

class ImageRotation:
  class KEYS(Model.KEYS):
    class TENSOR(Model.KEYS.TENSOR):
      SOURCE = 'source'
      X = 'xlors'
      Y = 'ylors'
      Z = 'zlors'
  
  def __init__(self, source, source_rotation):
    pass
  
