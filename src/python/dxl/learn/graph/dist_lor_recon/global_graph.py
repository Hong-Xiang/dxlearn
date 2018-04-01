from ..core import Graph
from typing import Iterable
import numpy as np
import tensorflow as tf
from ..core import DistributeGraphInfo, Host, Master, Barrier
from ..core import ThisHost
from ..model.on_collections import Summation
from ..core.utils import map_data

from .utils import constant_tensor, variable_tensor

class GlobalGraph(Graph):
  class KEYS(Graph.KEYS):
    class TENSOR(Graph.KEYS.TENSOR):
      X = 'x'
      X_BUFFER = 'x_buffer'
      X_INIT = 'x_init'
      INIT_OP = 'init_op'
      X_UPDATE = 'x_update'

  def make_tensors(self, x, ginfo):
    x, x_init = variable_tensor(x, 'x', ginfo)
    return {
        self.KEYS.TENSOR.X: x,
        self.KEYS.TENSOR.X_INIT: x_init,
        self.KEYS.TENSOR.X_BUFFER = [],
        self.KEYS.TENSOR.INIT_OP = [x_init]
    }

  def __init__(self, x, image_info,graph_info):
    self.image_info = image_info
    super().__init__(
        'global_graph',
        self.make_tensors(x, graph_info),
        graph_info=graph_info)

  def copy_x_to_local(self, host):
    x_cp, x_l = self.tensor(self.KEYS.TENSOR.X).copy_to(local_graph.host, True)
    return x_cp, x_l
  
  def add_local_result(self, x):
    self.tensors[self.KEYS.TENSOR.X_BUFFER].append(x)

  def merge_local_x(self):
    sm = Summation('summation', self.graph_info.update(name=None))
    TK = self.KEYS.TENSOR
    x_s = sm(self.tensor(TK.X_BUFFER))
    x_u = self.tensor(TK.X).assign(x_s)
    self.tensors[TK.X_UPDATE] = x_u
    return x_u

