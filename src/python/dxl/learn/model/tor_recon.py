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

ALL_AXIS = ['x', 'y', 'z']

ROTATIONS = {'x': [1, 2, 0], 'y': [0, 2, 1], 'z': [0, 1, 2]}
BACK_ROTATIONS = {'x': [2, 0, 1], 'y': [0, 2, 1], 'z': [2, 1, 0]}
ROTATIONS_IMAGE = {'x': [2, 0, 1], 'y': [1, 0, 2], 'z': [0, 1, 2]}
BACK_ROTATIONS_IMAGE = {'x': [1, 2, 0], 'y': [1, 0, 2], 'z': [0, 1, 2]}
KERNEL_WIDTH = np.sqrt(3.0 * 3.0 * np.pi)


def rotate(x, axis: str):
  if axis.lower() in ROTATIONS:
    return [x[i] for i in ROTATIONS[axis.lower()]]
  raise ValueError("Unsupported axis: {}.".format(axis))


def indices(a, offset=0):
  return [i + offset for i in ROTATIONS[a]]

class ReconStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            EFFICIENCY_MAP = 'efficiency_map'
            LORS_X = 'xlors'
            LORS_Y = 'ylors'
            LORS_Z = 'zlors'

    def __init__(self, name, image, efficiency_map,
                 grid, center, size,
                 xlors, ylors, zlors, graph_info):
        self.grid = grid
        self.center = center
        self.size = size
        self.eps = 1e-8
        super().__init__(name,
                         {self.KEYS.TENSOR.IMAGE: image,
                          self.KEYS.TENSOR.EFFICIENCY_MAP: efficiency_map,
                          self.KEYS.TENSOR.LORS_X: xlors,
                          self.KEYS.TENSOR.LORS_Y: ylors,
                          self.KEYS.TENSOR.LORS_Z: zlors
                          },
                         graph_info=graph_info)

    def kernel(self, inputs):
        img = self.tensor(self.KEYS.TENSOR.IMAGE)
        projections = Projection(
            'projection',
            img,
            self.grid,
            self.center,
            self.size,
            self.tensor(self.KEYS.TENSOR.LORS_X),
            self.tensor(self.KEYS.TENSOR.LORS_Y),
            self.tensor(self.KEYS.TENSOR.LORS_Z),
            self.graph_info.update(name=None))()
        backprojections = BackProjection(
            'backprojection',
            img,
            self.grid,
            self.center,
            self.size,
            self.tensor(self.KEYS.TENSOR.LORS_X),
            self.tensor(self.KEYS.TENSOR.LORS_Y),
            self.tensor(self.KEYS.TENSOR.LORS_Z),
            projections['x'],
            projections['y'],
            projections['z'],
            self.graph_info.update(name=None))()
        new_img = sum(map(lambda t: t.data, backprojections.values()))
        emap = self.tensor(self.KEYS.TENSOR.EFFICIENCY_MAP).data
        result = img.data / (emap + 1e-8) * new_img
        return Tensor(result, None, self.graph_info.update(name=None))





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
    imgz = inputs[self.KEYS.TENSOR.IMAGE].data
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
    # lors = {k: tf.transpose(lors[k]) for k in lors}

    # xlors = xlors[:, [1, 2, 0, 4, 5, 3]]
    # ylors = ylors[:, [0, 2, 1, 3, 5, 4]]

    model = 'tor'
    kernel_width = KERNEL_WIDTH

    def projection_axis(axis):
      print(lors[axis])
      print(imgs[axis])
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
    imgz = inputs[self.KEYS.TENSOR.IMAGE].data
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
        a: Tensor(v, None, self.graph_info.update(name=None))
        for a, v in backprojections.items()
    }



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

