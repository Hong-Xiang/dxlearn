from dxl.learn.core import Model, Tensor
from dxl.learn.model.tor_recon import Projection, BackProjection
import tensorflow as tf
import numpy as np
import warn

warn(DeprecationWarning())


class ReconStep(Model):
  class KEYS(Model.KEYS):
    class TENSOR(Model.KEYS.TENSOR):
      IMAGE = 'image'
      EFFICIENCY_MAP = 'efficiency_map'
      LORS_X = 'xlors'
      LORS_Y = 'ylors'
      LORS_Z = 'zlors'

  def __init__(self, name, image, efficiency_map, grid, center, size, xlors,
               ylors, zlors, graph_info):
    self.grid = grid
    self.center = center
    self.size = size
    self.eps = 1e-8
    super().__init__(
        name, {
            self.KEYS.TENSOR.IMAGE: image,
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