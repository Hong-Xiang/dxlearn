from typing import Iterable

import numpy as np
import tensorflow as tf

from ...core import Graph
from ...core.tensor import variable
from ...core.utils import logger, map_data
from ...model.on_collections import Summation
from .utils import constant_tensor, variable_tensor


class MasterGraph(Graph):
  class KEYS(Graph.KEYS):
    class TENSOR(Graph.KEYS.TENSOR):
      X = 'x'
      BUFFER = 'x_buffer'
      UPDATE = 'x_update'

  def __init__(self, x, nb_workers, graph_info=None, name='master_graph'):
    super().__init__(name, graph_info=graph_info)
    self._construct_x(x, nb_workers)
    self._construct_summation()
    self._debug_info()

  def _construct_x(self, x, nb_workers):
    x = variable(self.graph_info.update(name='x'), initializer=x)
    buffer = [
        variable(
            self.graph_info.update(name='buffer_{}'.format(i)),
            shape=x.shape,
            dtype=x.dtype) for i in range(nb_workers)
    ]
    self.get_tensor(self.KEYS.TENSOR.X, x)
    self.get_tensor(self.KEYS.TENSOR.BUFFER)

  def _construct_summation(self):
    sm = self.get_subgraph(
        lambda s: Summation('summation', s.graph_info.update(name=None)))
    x_s = sm(self.tensor(self.KEYS.TENSOR.BUFFER))
    x_u = self.tensor(self.KEYS.TENSOR.X).assign(x_s)
    self.tensors[self.KEYS.TENSOR.UPDATE] = x_u
    return x_u

  def _debug_info(self):
    logger.debug('Master graph constructed.')
    logger.debug('X: {}'.format(self.tensor(self.KEYS.TENSOR.X).data))
    logger.debug('BUFFER: {}'.format(
        list(map(lambda t: t.data, self.tensor(self.KEYS.TENSOR.BUFFER)))))
    logger.debug('UPDATE: {}'.format(
        self.tensor(self.KEYS.TENSOR.UPDATE).data))
