from .utils import constant_tensor
from ...core import Master, tf_tensor, Graph, variable
import tensorflow as tf


class WorkerGraphBase(Graph):
  class KEYS(Graph.KEYS):
    class TENSOR(Graph.KEYS.TENSOR):
      X = 'x'
      UPDATE = 'update'
      RESULT = 'result'

  def __init__(self, global_graph, task_index=None, graph_info=None,
               name=None):
    if task_index is None:
      task_index = self.global_graph.host.task_index
    self.task_index = task_index
    if name is None:
      name = 'worker_graph_{}'.format(self.task_index)
    self.global_graph = global_graph
    super().__init__(name, graph_info=graph_info)
    self._construct_x()
    self._construct_x_result()
    self._construct_x_update()

  def _construct_x(self):
    x_global = self.global_graph.tensor(self.global_graph.KEYS.TENSOR.X)
    self.tensors[self.KEYS.TENSOR.X] = x_global

  def _construct_x_result(self):
    self.tensors[self.KEYS.TENSOR.RESULT] = self.tensor(self.KEYS.TENSOR.X)

  def _construct_x_update(self):
    x_buffers = self.global_graph.tensor(self.global_graph.KEYS.TENSOR.BUFFER)
    x_buffer = x_buffers[self.task_index]
    x_u = x_buffer.assign(self.tensor(self.KEYS.TENSOR.RESULT))
    self.tensors[self.KEYS.TENSOR.UPDATE] = x_u

  def init_op(self):
    """
        This method is intend to be called in global_graph.init_op. 
        """
    op = tf.no_op()
    self.tensors[self.KEYS.TENSOR.INIT_OP] = op
    return op


class WorkerGraphLOR(WorkerGraphBase):
  class KEYS(WorkerGraphBase.KEYS):
    class TENSOR(WorkerGraphBase.KEYS.TENSOR):
      EFFICIENCY_MAP = 'efficiency_map'
      LORS = 'lors'
      INIT = 'init'

  def __init__(self, global_graph, image_info, efficiency_map_shape,
               lors_shape, task_index, graph_info):
    super().__init__(global_graph, task_index, graph_info)
    self.image_info = image_info
    self._construct_inputs(efficiency_map_shape, lors_shape)

  def _construct_inputs(self, map_shape, lors_shape):
    KT = self.KEYS.TENSOR
    tid = self.host.task_index
    self.tensors[KT.EFFICIENCY_MAP] = variable(
        self.graph_info.update(name='effmap_{}'.format(tid)),
        None,
        map_shape,
        tf.float32)
    self.tensors[KE.LORS] = {
        a: variable(
            self.graph_info.update(name='lor_{}_{}'.format(a, tid)),
            None,
            lors_shape[a],
            tf.float32)
        for a in lors_shape
    }

  def bind(self, efficiency_map, lors):
    map_assign = self.tensor(
        self.KEYS.TENSOR.EFFICIENCY_MAP).assign(efficiency_map)
    lors_assign = [
        self.tensor(self.KEYS.TENSOR.LORS)[a].assign(lors[a]) for a in lors
    ]
    with tf.control_dependencies([map_assign] + lors_assign):
      return tf.no_op()

  def _construct_x_result(self):
    KT = self.KEYS.TENSOR
    from ...model.tor_recon import ReconStep
    return ReconStep(
        'recon_step_{}'.format(self.host.task_index),
        self.tensor(KT.X_COPY_FROM_GLOBAL),
        self.tensor(KT.EFFICIENCY_MAP),
        self.tensor(KT.XLORS),
        self.image_info,
        self.graph_info.update(name=None))()
