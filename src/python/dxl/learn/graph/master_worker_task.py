from ..core import (DistributeGraphInfo, Graph, Host, MasterHost, ThisHost,
                    ThisSession, make_cluster, make_distribute_session)

from ..core.distribute import JOB_NAME


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
                 name=None,
                 graph_info=None,
                 config=None,
                 *,
                 job=None,
                 task_index=None,
                 cluster_config=None):
        KC = self.KEYS.CONFIG
        if name is None:
            name = 'master_worker_task'
        if config is None:
            config = {}
        config.update({
            KC.CLUSTER: cluster_config,
            KC.JOB: job,
            KC.TASK_INDEX: task_index
        })
        super().__init__(name, graph_info=graph_info, config=config)
        self.hosts = {JOB_NAME.MASTER: None, JOB_NAME.WORKER: []}
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

    def default_info(self):
        return DistributeGraphInfo(self.name, self.name,
                                   Host(
                                       self.config(self.KEYS.CONFIG.JOB),
                                       self.config(
                                           self.KEYS.CONFIG.TASK_INDEX)))

    @property
    def job(self):
        return self.config(self.KEYS.CONFIG.JOB)

    @property
    def nb_workers(self):
        return self.config(self.KEYS.CONFIG.NB_WORKERS)

    @property
    def job(self):
        return self.config(self.KEYS.CONFIG.JOB)

    @property
    def task_index(self):
        return self.config(self.KEYS.CONFIG.TASK_INDEX)

    def _cluster_init(self):
        """
        Create cluster to run this task, this function should be called before:
        - self.nb_workers()
        """

        self.config.update(
            self.KEYS.CONFIG.NB_WORKERS,
            len(self.config(self.KEYS.CONFIG.CLUSTER)[JOB_NAME.WORKER]))
        KC = self.KEYS.CONFIG
        make_cluster(
            self.config(KC.CLUSTER), self.config(KC.JOB),
            self.config(KC.TASK_INDEX), Host(JOB_NAME.MASTER, 0))
        self.hosts[JOB_NAME.MASTER] = MasterHost.host()
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
        make_distribute_session()

    # def run_step_of_this_host(self, name):
    #     if ThisHost.is_master():
    #         ThisSession.run(self.steps[name][JOB_NAME.MASTER])
    #     else:
    #         ThisSession.run(
    #             self.steps[name][JOB_NAME.WORKER][ThisHost.host().task_index])

    # def worker_graph_on(self, host):
    #     return self.subgraph(JOB_NAME.WORKER)[host.task_index]

    # def graph_on_this_host(self):
    #     host = ThisHost.host()
    #     if ThisHost.is_master():
    #         return self.master_graph
    #     else:
    #         for g in self.worker_graphs:
    #             if g.graph_info.host == host:
    #                 return g
    #     raise KeyError("No local graph for {}.{} found".format(
    #         host.job, host.task_index))

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


__all__ = ['MasterWorkerTask', 'JOB_NAME']
