from ..core import (DistributeGraphInfo, Graph, Host, MasterHost, ThisHost,
                    ThisSession, make_cluster, make_distribute_session)

from ..core.distribute import JOB_NAME

import warnings


class MasterWorkerTaskBase(Graph):
    """
    Helper class of managing distribute task with Master-Multiple Worker model.

    `self.config(KC.CLUSTER)` a ClusterSpec Object.
    """

    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            pass

        class CONFIG(Graph.KEYS.CONFIG):
            CLUSTER = 'cluster'
            JOB = 'job'
            TASK_INDEX = 'task_index'
            NB_WORKERS = 'nb_workers'

        class SUBGRAPH(Graph.KEYS.SUBGRAPH):
            MASTER = JOB_NAME.MASTER
            WORKER = JOB_NAME.WORKER

    def __init__(self,
                 info=None,
                 config=None,
                 *,
                 job=None,
                 task_index=None,
                 cluster_config=None):
        KC = self.KEYS.CONFIG
        if info is None:
            info = 'master_worker_task'
        if config is None:
            config = {}
        config.update({
            KC.CLUSTER: cluster_config,
            KC.JOB: job,
            KC.TASK_INDEX: task_index
        })

        super().__init__(info, config=config)
        self.hosts = {JOB_NAME.MASTER: None, JOB_NAME.WORKER: []}
        self.is_cluster_init = False
        self._cluster_init()
        self._make_master_graph()
        self._make_worker_graphs()
        self._make_barriers()

    @classmethod
    def default_config(cls):
        from ..core.distribute import DEFAULT_CLUSTER_CONFIG
        return {
            cls.KEYS.CONFIG.CLUSTER: DEFAULT_CLUSTER_CONFIG,
            cls.KEYS.CONFIG.TASK_INDEX: 0
        }

    def default_info(self, name):
        return DistributeGraphInfo(name, name,
                                   Host(
                                       self.config(self.KEYS.CONFIG.JOB),
                                       self.config(
                                           self.KEYS.CONFIG.TASK_INDEX)))

    @property
    def job(self):
        return self.config(self.KEYS.CONFIG.JOB)

    @property
    def nb_workers(self):
        return len(self.config(self.KEYS.CONFIG.CLUSTER)[JOB_NAME.WORKER])

    @property
    def task_index(self):
        return self.config(self.KEYS.CONFIG.TASK_INDEX)

    def master_host(self):
        if not self.is_cluster_init:
            return Host(JOB_NAME.MASTER, 0)
        return MasterHost.host()

    def _make_cluster_on_backend(self):
        make_cluster(
            self.config(self.KEYS.CONFIG.CLUSTER), self.job, self.task_index,
            self.master_host())
        self.is_cluster_init = True

    def _cluster_init(self):
        """
        Create cluster to run this task, this function should be called before:
        - self.nb_workers()
        """
        self._make_cluster_on_backend()
        self.hosts[JOB_NAME.MASTER] = self.master_host()
        self.hosts[JOB_NAME.WORKER] = [
            Host(JOB_NAME.WORKER, i) for i in range(self.nb_workers)
        ]

    def _make_master_graph(self):
        """
        User might want to overwrite this function.
        """
        pass

    def _make_worker_graphs(self):
        """
        User might want to overwrite this function.
        """
        pass

    def _make_barriers(self):
        """
        User might want to overwrite this function.
        """
        pass

    def make_session(self):
        warnings.warn(
            DeprecationWarning(
                "Directly use dxl.learn.make_distribute_session instead."))
        make_distribute_session()

    @classmethod
    def worker_only(cls, func):
        def call(*args, **kwargs):
            if not ThisHost.is_master():
                return func(*args, **kwargs)
            return None

        return call

    @classmethod
    def master_only(cls, func):
        def call(*args, **kwargs):
            if ThisHost.is_master():
                return func(*args, **kwargs)
            return None

        return call


__all__ = ['MasterWorkerTaskBase', 'JOB_NAME']
