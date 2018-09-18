from dxl.learn.graph.base import Graph
from ..distribute import (DistributeGraphInfo, Host, Master, ThisHost,
                          make_master_worker_cluster, make_distribute_session,
                          JOB_NAME, MasterWorkerCluster)

from functools import wraps


class AbstractMasterWorkerGraph(Graph):
    pass


class MasterWorkerTaskBase(Graph):
    """
    Helper class of managing distribute task with Master-Multiple Worker model.

    User need to implement two methods:
    `self._make_master_graph()`
    `self._make_worker_graph(task_index)`
    """

    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            CLUSTER = 'cluster'
            JOB = 'job'
            TASK_INDEX = 'task_index'
            NB_WORKERS = 'nb_workers'

        class GRAPH(Graph.KEYS.GRAPH):
            MASTER = JOB_NAME.MASTER
            WORKER = JOB_NAME.WORKER

    def __init__(self,
                 info=None,
                 config=None,
                 tensors=None,
                 graphs=None,
                 *,
                 job=None,
                 task_index=None,
                 cluster=None):
        if not isinstance(cluster, MasterWorkerCluster):
            raise TypeError(
                "Invalid cluster type, required {}, got {}.".format(
                    MasterWorkerCluster, type(cluster)))
        KC = self.KEYS.CONFIG
        if info is None:
            info = 'master_worker_task'
        config = self._parse_input_config(config, {
            KC.JOB: job,
            KC.TASK_INDEX: task_index
        })

        self._cluster = cluster
        super().__init__(info, config, tensors, graphs)

    @classmethod
    def _default_config(cls):
        from ..distribute import MasterWorkerClusterSpec
        return {
            cls.KEYS.CONFIG.CLUSTER:
                MasterWorkerClusterSpec.make_local_cluster_config(2),
            cls.KEYS.CONFIG.TASK_INDEX:
                0
        }

    def _default_info(self, name):
        host = Host(
            self.config(self.KEYS.CONFIG.JOB),
            self.config(self.KEYS.CONFIG.TASK_INDEX))
        return DistributeGraphInfo(name, host, name)

    @property
    def job(self) -> str:
        return self.this_host().job

    @property
    def task_index(self) -> int:
        return self.this_host().task_index

    @property
    def nb_workers(self) -> int:
        return self._cluster.nb_workers

    def master(self) -> Host:
        return self._cluster.master()

    def worker(self, task_index) -> Host:
        return self._cluster.worker(task_index)

    def this_host(self) -> Host:
        return self._cluster.host(
            self.config(self.KEYS.CONFIG.JOB),
            self.config(self.KEYS.CONFIG.TASK_INDEX))

    def _cluster_init(self):
        """
        Create cluster to run this task, this function should be called before:
        - self.nb_workers()
        """
        self._cluster = make_master_worker_cluster(
            self.config(self.KEYS.CONFIG.CLUSTER),
            self.config(self.KEYS.CONFIG.JOB),
            self.config(self.KEYS.CONFIG.TASK_INDEX))

    def master_info(self, name, scope=None):
        return DistributeGraphInfo(name, self.master(), scope)

    def worker_info(self, task_index, name, scope=None):
        return DistributeGraphInfo(name, self.worker(task_index), scope)

    @classmethod
    def worker_only(cls, func):
        @wraps(func)
        def call(*args, **kwargs):
            if not ThisHost.is_master():
                return func(*args, **kwargs)
            return None

        return call

    @classmethod
    def master_only(cls, func):
        @wraps(func)
        def call(*args, **kwargs):
            if ThisHost.is_master():
                return func(*args, **kwargs)
            return None

        return call


__all__ = ['MasterWorkerTaskBase', 'JOB_NAME']
