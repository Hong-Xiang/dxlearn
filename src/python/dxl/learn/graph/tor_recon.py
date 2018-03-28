from ..core import Graph
from typing import Iterable
import numpy as np
import tensorflow as tf
from ..core import VariableInfo, TensorVariable, TensorNumpyNDArray, DistributeGraphInfo, Host, Master, Barrier
from ..core import ThisHost
from ..model.tor_recon import ProjectionSplitter, ReconStep
from ..model.on_collections import Summation
from ..core.utils import map_data


class GlobalGraph(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            XLORS = 'xlors'
            YLORS = 'ylors'
            ZLORS = 'zlors'
            GRID = 'grid'
            CENTER = 'center'
            SIZE = 'size'
            # SYSTEM_MATRIX = 'system_matrix'
            EFFICIENCY_MAP = 'efficiency_map'
            X_BUFFER = 'x_buffer'
            X_BUFFER_TARGET = 'x_buffer_target'
            X_INIT = 'x_init'
            XLORS_SPLIT = 'xlors_split'
            YLORS_SPLIT = 'ylors_split'
            ZLORS_SPLIT = 'zlors_split'
            # SYSTEM_MATRIX_SPLIT = 'system_matrix_split'
            LOCAL_INIT_OPS = 'local_init_ops'
            INIT_OP = 'init_op'
            X_UPDATE = 'x_update'

    def make_tensors(self, x: np.ndarray, grid: np.ndarray,
                     center: np.ndarray, size: np.ndarray,
                     xlors: np.ndarray, ylors: np.ndarray,
                     zlors: np.ndarray, em: np.ndarray,
                     graph_info: DistributeGraphInfo):
        x = x.astype(np.float32)
        grid = grid.astype(np.int32)
        center = center.astype(np.float32)
        size = size.astype(np.float32)
        xlors = xlors.astype(np.float32)
        ylors = ylors.astype(np.float32)
        zlors = zlors.astype(np.float32)
        # sm = sm.astype(np.float32)
        em = em.astype(np.float32)
        x_var_info = VariableInfo(None, x.shape, tf.float32)
        x_t = TensorVariable(x_var_info, graph_info.update(name='x_t'))
        x_init = x_t.assign(TensorNumpyNDArray(x, None,
                                               x_t.graph_info.update(name='x_init')))
        grid_t = TensorNumpyNDArray(grid, None,
                                    x_t.graph_info.update(name='grid'))
        center_t = TensorNumpyNDArray(center, None,
                                      x_t.graph_info.update(name='center'))
        size_t = TensorNumpyNDArray(size, None,
                                    x_t.graph_info.update(name='size'))
        em_t = TensorNumpyNDArray(em, None,
                                  x_t.graph_info.update(name='effmap'))
        xlors_t = TensorNumpyNDArray(xlors, None,
                                     x_t.graph_info.update(name='xlors'))
        ylors_t = TensorNumpyNDArray(ylors, None,
                                     x_t.graph_info.update(name='ylors'))
        zlors_t = TensorNumpyNDArray(zlors, None,
                                     x_t.graph_info.update(name='zlors'))
        # y_t = TensorNumpyNDArray(y, None, x_t.graph_info.update(name='y'))
        # sm_t = TensorNumpyNDArray(sm, None,
        #                           x_t.graph_info.update(name='system_matrix'))
        return {
            self.KEYS.TENSOR.X: x_t,
            self.KEYS.TENSOR.X_INIT: x_init,
            self.KEYS.TENSOR.XLORS: xlors_t,
            self.KEYS.TENSOR.YLORS: ylors_t,
            self.KEYS.TENSOR.ZLORS: zlors_t,
            self.KEYS.TENSOR.GRID: grid_t,
            self.KEYS.TENSOR.CENTER: center_t,
            self.KEYS.TENSOR.SIZE: size_t,
            # self.KEYS.TENSOR.Y: y_t,
            # self.KEYS.TENSOR.SYSTEM_MATRIX: sm_t,
            self.KEYS.TENSOR.EFFICIENCY_MAP: em_t,
            self.KEYS.TENSOR.X_BUFFER: [],
            self.KEYS.TENSOR.X_BUFFER_TARGET: [],
            self.KEYS.TENSOR.LOCAL_INIT_OPS: [],
        }

    def __init__(self, x: np.ndarray, grid: np.ndarray,
                 center: np.ndarray, size: np.ndarray,
                 xlors: np.ndarray, ylors: np.ndarray,
                 zlors: np.ndarray, em: np.ndarray, graph_info):
        super().__init__('global_graph',
                         self.make_tensors(
                             x, grid, center, size,
                             xlors, ylors, zlors, em, graph_info),
                         graph_info=graph_info)

    def split(self, nb_workers):
        spt = ProjectionSplitter('splitter', nb_workers,
                                 self.graph_info.update(name=None))

        xlors_ts = spt(self.tensor(self.KEYS.TENSOR.XLORS))
        ylors_ts = spt(self.tensor(self.KEYS.TENSOR.YLORS))
        zlors_ts = spt(self.tensor(self.KEYS.TENSOR.ZLORS))

        xlors_ts = [xlors_ts['slice_{}'.format(i)] for i in range(nb_workers)]
        ylors_ts = [ylors_ts['slice_{}'.format(i)] for i in range(nb_workers)]
        zlors_ts = [zlors_ts['slice_{}'.format(i)] for i in range(nb_workers)]

        self.tensors[self.KEYS.TENSOR.XLORS_SPLIT] = xlors_ts
        self.tensors[self.KEYS.TENSOR.YLORS_SPLIT] = ylors_ts
        self.tensors[self.KEYS.TENSOR.ZLORS_SPLIT] = zlors_ts
        # y_ts = spt(self.tensor(self.KEYS.TENSOR.Y))
        # sm_ts = spt(self.tensor(self.KEYS.TENSOR.SYSTEM_MATRIX))

        # y_ts = [y_ts['slice_{}'.format(i)] for i in range(nb_workers)]
        # sm_ts = [sm_ts['slice_{}'.format(i)] for i in range(nb_workers)]
        # self.tensors[self.KEYS.TENSOR.Y_SPLIT] = y_ts
        # self.tensors[self.KEYS.TENSOR.SYSTEM_MATRIX_SPLIT] = sm_ts

    def copy_to_local(self, host: Host) -> 'LocalGraph':
        tid = host.task_index
        x_cp, x_l = self.tensor(self.KEYS.TENSOR.X).copy_to(host, True)
        grid_cp, grid_l = self.tensor(
            self.KEYS.TENSOR.GRID).copy_to(host, True)
        center_cp, center_l = self.tensor(
            self.KEYS.TENSOR.CENTER).copy_to(host, True)
        size_cp, size_l = self.tensor(
            self.KEYS.TENSOR.SIZE).copy_to(host, True)
        # y_cp, y_l = self.tensor(self.KEYS.TENSOR.Y_SPLIT)[tid].copy_to(
        #     host, True)
        # sm_cp, sm_l = self.tensor(self.KEYS.TENSOR.SYSTEM_MATRIX_SPLIT)[tid].copy_to(
        #     host, True)
        xlors_cp, xlors_l = self.tensor(self.KEYS.TENSOR.XLORS_SPLIT)[tid].copy_to(
            host, True)
        ylors_cp, ylors_l = self.tensor(self.KEYS.TENSOR.YLORS_SPLIT)[tid].copy_to(
            host, True)
        zlors_cp, zlors_l = self.tensor(self.KEYS.TENSOR.ZLORS_SPLIT)[tid].copy_to(
            host, True)
        em_cp, em_l = self.tensor(self.KEYS.TENSOR.EFFICIENCY_MAP).copy_to(
            host, True)
        self.tensors[self.KEYS.TENSOR.LOCAL_INIT_OPS] += [x_cp, grid_cp,
                                                          center_cp, size_cp,
                                                          xlors_cp, ylors_cp,
                                                          zlors_cp, em_cp]
        return LocalGraph(tid, x_l, em_l, grid_l, center_l, size_l,  xlors_l, ylors_l, zlors_l,
                          x_cp, self.graph_info.update(name=None, host=host))

    def init_op(self, local_graphs: Iterable['LocalGraph']):
        master_host = Master.master_host()
        hosts = [g.graph_info.host for g in local_graphs]
        TK = self.KEYS.TENSOR
        with tf.control_dependencies([self.tensor(TK.X_INIT).data]):
            with tf.control_dependencies(map_data(self.tensor(TK.LOCAL_INIT_OPS))):
                _op = tf.no_op()
        init_barrier = Barrier('init', [master_host], hosts, [_op])

        master_op = init_barrier.barrier(master_host)
        workers_op = [init_barrier.barrier(h) for h in hosts]
        return master_op, workers_op

    def x_update_by_merge(self):
        sm = Summation('summation', self.graph_info.update(name=None))
        TK = self.KEYS.TENSOR
        # op_copy_back = []
        # for b, bt in zip(self.tensor(TK.X_BUFFER), self.tensor(TK.X_BUFFER_TARGET)):
        # op_copy_back.append(b.assign(bt))
        # with tf.control_dependencies(map_data(op_copy_back)):
        x_s = sm(self.tensor(TK.X_BUFFER))
        x_u = self.tensor(TK.X).assign(x_s)
        self.tensors[TK.X_UPDATE] = x_u
        return x_u

    def recon_step(self, local_graphs: Iterable['LocalGraph'], worker_hosts):
        recon_local = [g.tensor(g.KEYS.TENSOR.X_UPDATE) for g in local_graphs]
        master_host = Master.master_host()
        calculate_barrier = Barrier('calculate', worker_hosts, [master_host],
                                    task_lists=[[r] for r in recon_local])
        # import pdb
        # pdb.set_trace()
        TK = self.KEYS.TENSOR
        master_barrier = calculate_barrier.barrier(master_host)
        worker_barriers = [calculate_barrier.barrier(h) for h in worker_hosts]
        with tf.control_dependencies([master_barrier]):
            _op = self.x_update_by_merge()
        return _op.data, worker_barriers

    def merge_step(self, master_recon_op, worker_recon_ops,  worker_hosts):
        merge_barrier = Barrier('merge', [Master.master_host()],
                                worker_hosts, [[master_recon_op]])
        master_op = merge_barrier.barrier(Master.master_host())
        workers_ops = []
        for op, h in zip(worker_recon_ops, worker_hosts):
            with tf.control_dependencies([op, merge_barrier.barrier(h)]):
                workers_ops.append(tf.no_op())
        return master_op, workers_ops

        # l_ops = []
        # for h in worker_hosts:
        #     with tf.control_dependencies([merge_barrier.barrier(h)]):
        #         l_ops.append(tf.no_op())
        # return gop, l_ops


class LocalGraph(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            EFFICIENCY_MAP = 'efficiency_map'
            GRID = 'grid'
            CENTER = 'center'
            SIZE = 'size'
            XLORS = 'xlors'
            YLORS = 'ylors'
            ZLORS = 'zlors'
            # Y = 'y'
            # SYSTEM_MATRIX = 'system_matrix'
            X_COPY_FROM_GLOBAL = 'x_copy_from_global'
            X_COPY_TO_GLOBAL = 'x_copy_to_global'
            X_UPDATE = 'x_update'
            X_RESULT = 'x_result'
            X_GLOBAL_BUFFER = 'x_global_buffer'

    def __init__(self, tid, x, em, grid, center, size,
                 xlors, ylors, zlors, x_copy, graph_info):
        self.tid = tid
        name = 'local_graph_{}'.format(tid)
        super().__init__(name, {
            self.KEYS.TENSOR.X: x,
            self.KEYS.TENSOR.EFFICIENCY_MAP: em,
            self.KEYS.TENSOR.GRID: grid,
            self.KEYS.TENSOR.CENTER: center,
            self.KEYS.TENSOR.SIZE: size,
            # self.KEYS.TENSOR.Y: y,
            self.KEYS.TENSOR.XLORS: xlors,
            self.KEYS.TENSOR.YLORS: ylors,
            self.KEYS.TENSOR.ZLORS: zlors,
            # self.KEYS.TENSOR.SYSTEM_MATRIX: sm,

            self.KEYS.TENSOR.X_COPY_FROM_GLOBAL: x_copy
        }, graph_info=graph_info)

    def copy_to_global(self, global_graph: GlobalGraph):
        gg = global_graph
        x = self.tensor(self.KEYS.TENSOR.X)
        x_cp, x_b = x.copy_to(Master.master_host(), True)
        gg.tensors[global_graph.KEYS.TENSOR.X_BUFFER].append(x_b)
        gg.tensors[global_graph.KEYS.TENSOR.X_BUFFER_TARGET].append(x)
        self.tensors[self.KEYS.TENSOR.X_COPY_TO_GLOBAL] = x_cp
        self.tensors[self.KEYS.TENSOR.X_GLOBAL_BUFFER] = x_b
        return x_cp

    def recon_local(self):
        x_n = ReconStep('recon_step_{}'.format(self.tid),
                        self.tensor(self.KEYS.TENSOR.X_COPY_FROM_GLOBAL),
                        # self.tensor(self.KEYS.TENSOR.Y),
                        self.tensor(self.KEYS.TENSOR.EFFICIENCY_MAP),
                        self.tensor(self.KEYS.TENSOR.GRID),
                        self.tensor(self.KEYS.TENSOR.CENTER),
                        self.tensor(self.KEYS.TENSOR.SIZE),
                        self.tensor(self.KEYS.TENSOR.XLORS),
                        self.tensor(self.KEYS.TENSOR.YLORS),
                        self.tensor(self.KEYS.TENSOR.ZLORS),
                        # self.tensor(self.KEYS.TENSOR.SYSTEM_MATRIX),
                        self.graph_info.update(name=None))()
        self.tensors[self.KEYS.TENSOR.X_RESULT] = x_n
        x_u = self.tensor(self.KEYS.TENSOR.X_GLOBAL_BUFFER).assign(x_n)
        # x_u = self.tensor(self.KEYS.TENSOR.X).assign(x_n)
        self.tensors[self.KEYS.TENSOR.X_UPDATE] = x_u
        return x_u
