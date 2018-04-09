from .utils import constant_tensor
from ...core import Master, tf_tensor, Graph, variable, Tensor
import tensorflow as tf


class WorkerGraphBase(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            UPDATE = 'update'
            RESULT = 'result'

    def __init__(self, global_graph, task_index=None, graph_info=None,
                 name=None):
        self.global_graph = global_graph
        if task_index is None:
            task_index = self.global_graph.host.task_index
        self.task_index = task_index
        if name is None:
            name = 'worker_graph_{}'.format(self.task_index)

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
        x_buffers = self.global_graph.tensor(
            self.global_graph.KEYS.TENSOR.BUFFER)
        x_buffer = x_buffers[self.task_index]
        x_u = x_buffer.assign(self.tensor(self.KEYS.TENSOR.RESULT))
        self.tensors[self.KEYS.TENSOR.UPDATE] = x_u


class WorkerGraphLOR(WorkerGraphBase):
    class KEYS(WorkerGraphBase.KEYS):
        class TENSOR(WorkerGraphBase.KEYS.TENSOR):
            EFFICIENCY_MAP = 'efficiency_map'
            LORS = 'lors'
            INIT = 'init'

    def __init__(self,
                 master_graph,
                 image_info,
                 lors_shape,
                 task_index,
                 graph_info=None,
                 name=None):
        self.image_info = image_info
        self.lors_shape = lors_shape
        super().__init__(master_graph, task_index, graph_info, name=name)

    def _construct_inputs(self):
        KT = self.KEYS.TENSOR
        self.tensors[KT.EFFICIENCY_MAP] = variable(
            self.graph_info.update(name='effmap_{}'.format(self.task_index)),
            None,
            self.tensor(self.KEYS.TENSOR.X).shape,
            tf.float32)
        self.tensors[KT.LORS] = {
            a: variable(
                self.graph_info.update(
                    name='lor_{}_{}'.format(a, self.task_index)),
                None,
                self.lors_shape[a],
                tf.float32)
            for a in self.lors_shape
        }
        self.tensors[KT.INIT] = Tensor(
            tf.no_op(), None, self.graph_info.update(name='init_no_op'))

    def assign_efficiency_map_and_lors(self, efficiency_map, lors):
        map_assign = self.tensor(
            self.KEYS.TENSOR.EFFICIENCY_MAP).assign(efficiency_map)
        lors_assign = [
            self.tensor(self.KEYS.TENSOR.LORS)[a].assign(lors[a]) for a in lors
        ]
        with tf.control_dependencies(
                [map_assign.data] + [a.data for a in lors_assign]):
            init = tf.no_op()
        init = Tensor(init, None, self.graph_info.update(name='init'))
        self.tensors[self.KEYS.TENSOR.INIT] = init

    def _construct_x_result(self):
        self._construct_inputs()
        KT = self.KEYS.TENSOR
        from ...model.tor_step import TorStep
        x_res = TorStep(
            'recon_step_{}'.format(self.task_index),
            self.tensor(KT.X, is_required=True),
            self.tensor(KT.EFFICIENCY_MAP, is_required=True),
            self.image_info.grid,
            self.image_info.center,
            self.image_info.size,
            self.tensor(self.KEYS.TENSOR.LORS)['x'],
            self.tensor(self.KEYS.TENSOR.LORS)['y'],
            self.tensor(self.KEYS.TENSOR.LORS)['z'],
            self.graph_info.update(name=None))()
        self.tensors[self.KEYS.TENSOR.RESULT] = x_res
