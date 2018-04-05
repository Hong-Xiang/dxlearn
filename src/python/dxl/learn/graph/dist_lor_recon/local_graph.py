from .utils import constant_tensor
from ...core import Master, tf_tensor, Graph


class LocalGraphBase(Graph):
  class KEYS(Graph.KEYS):
    class TENSOR(Graph.KEYS.TENSOR):
      X = 'x'
      X_COPY_FROM_GLOBAL = 'x_copy_from_global'
      X_UPDATE = 'x_update'
      INIT_OP = 'init_op'

  def __init__(self, global_graph, host, image_info, addtional_tensors,
               graph_info):
    self.host = host
    self.image_info = image_info
    self.global_graph = global_graph
    self.recon_step_maker = recon_step_maker
    name = 'local_graph_{}'.format(host.task_index)
    x, x_cp = global_graph.copy_to_local(host)
    input_tensors = {
        self.KEYS.TENSOR.X: x,
        self.KEYS.TENSOR.X_COPY_FROM_GLOBAL: x_cp
    }
    input_tensors.update(addtional_tensors)
    super().__init__(name, input_tensors, graph_info=graph_info)

  def make_x_result(self):
    raise NotImplementedError

  def make_recon_step(self):
    KT = self.KEYS.TENSOR

    x_cp, x_buffer = x_n.copy_to(Master.master_host(), True)
    self.tensors[self.KEYS.TENSOR.X_UPDATE] = x_cp
    self.global_graph.add_local_result(x_buffer)
    return x_cp

  def init_op(self):
    """
    This method is intend to be called in global_graph.init_op. 
    """
    op = tf.no_op()
    self.tensors[self.KEYS.TENSOR.INIT_OP] = op
    return op


class LocalGraphLOR(LocalGraphBase):
  class KEYS(Graph.KEYS):
    class TENSOR(Graph.KEYS.TENSOR):
      EFFICIENCY_MAP = 'efficiency_map'
      LORS = 'lors'

  def __init__(self, global_graph, host, image_info, efficiency_map, lors,
               graph_info):
    addtional_tensors = {
        self.KEYS.TENSOR.EFFICIENCY_MAP: constant_tensor(efficiency_map),
        self.KEYS.TENSOR.LORS: {a: constant_tensor(lors[a])
                                for a in lors}
    }
    super().__init__(global_graph, host, image_info, addtional_tensors,
                     graph_info)

  def make_x_result(self):
    from ...model.tor_recon import ReconStep
    return ReconStep(
        'recon_step_{}'.format(self.host.task_index),
        self.tensor(KT.X_COPY_FROM_GLOBAL),
        self.tensor(KT.EFFICIENCY_MAP),
        self.tensor(KT.XLORS),
        self.image_info,
        self.graph_info.update(name=None))()
