from ..core import (DEFAULT_CLUSTER_CONFIG, JOB_NAME, DistributeGraphInfo,
                    Graph, Host, MasterHost, ThisHost, ThisSession,
                    make_cluster)


class MasterWorkerTask(Graph):
    """
    Helper class of managing distribute task with Master-Multiple Worker model.
    """

    class KEYS:
        class STEPS:
            """
            Names of steps.
            """
            pass

        class CONFIG(Graph.KEYS.CONFIG):
            CLUSTER = 'cluster'
            JOB = 'job'
            TASK_INDEX = 'task_index'

        class SUBGRAPH(Graph.KEYS.SUBGRAPH):
            MASTER = JOB_NAME.MASTER
            WORKER = JOB_NAME.WORKER

    @classmethod
    def default_config(cls):
        return {cls.KEYS.CONFIG.CLUSTER: DEFAULT_CLUSTER_CONFIG}

    def __init__(self, job, task_index, distribute_config=None, name=None):
        KC = self.KEYS.CONFIG
        if name is None:
            name = 'distribute_task'
        super().__init__(
            name,
            config={
                KC.CLUSTER: distribute_config,
                KC.JOB: job,
                KC.TASK_INDEX: task_index
            })
        self.subgraphs[JOB_NAME.MASTER] = None
        self.subgraphs[JOB_NAME.WORKER] = []
        self.hosts = {JOB_NAME.MASTER: None, JOB_NAME.WORKER: []}
        self.steps = {}
        self.cluster_init()
        self.subgraph_infos = {JOB_NAME.MASTER: None, JOB_NAME.WORKER: []}

    def nb_workers(self):
        return len(self.config(self.KEYS.CONFIG.CLUSTER)[JOB_NAME.WORKER])

    def cluster_init(self):
        """
        Create cluster to run this task, this function should be called before:
        - self.nb_workers()
        """
        KC = self.KEYS.CONFIG
        make_cluster(
            self.config(KC.CLUSTER), self.config(KC.JOB),
            self.config(KC.TASK_INDEX), Host(JOB_NAME.MASTER, 0))
        self.hosts[JOB_NAME.MASTER] = MasterHost.host()
        self.hosts[JOB_NAME.WORKER] = [
            Host(JOB_NAME.WORKER, i) for i in range(self.nb_workers())
        ]
        self.subgraph_infos[JOB_NAME.MASTER] = DistributeGraphInfo(
            None, None, False, self.hosts[JOB_NAME.MASTER])
        self.subgraph_infos[JOB_NAME.WORKER] = [
            DistributeGraphInfo(None, None, False, h)
            for h in self.hosts[JOB_NAME.WORKER]
        ]

    def add_master_graph(self, g):
        self.subgraphs[JOB_NAME.MASTER] = g

    def add_worker_graph(self, g):
        self.subgraphs[JOB_NAME.WORKER].append(g)

    def add_step(self, name, master_op, worker_ops):
        """
        Add step to run, add a step dict with
        {'mater': mater_op, 'worker': [worker_ops]}
        """
        self.steps[name] = {
            JOB_NAME.MASTER: master_op,
            JOB_NAME.WORKER: worker_ops
        }

    def run_step_of_this_host(self, name):
        if ThisHost.is_master():
            ThisSession.run(self.steps[name][JOB_NAME.MASTER])
        else:
            ThisSession.run(
                self.steps[name][JOB_NAME.WORKER][ThisHost.host().task_index])

    def worker_graph_on(self, host):
        return self.subgraph(JOB_NAME.WORKER)[host.task_index]

    def graph_on_this_host(self):
        host = ThisHost.host()
        if ThisHost.is_master():
            return self.master_graph
        else:
            for g in self.worker_graphs:
                if g.graph_info.host == host:
                    return g
        raise KeyError("No local graph for {}.{} found".format(
            host.job, host.task_index))

    def ginfo_master(self):
        return self.master_graph_info

    def ginfo_worker(self, task_index):
        return self.worker_graph_infos[task_index]

    def ginfo_this(self):
        """
        Helper function to provide graph_info of ThisHost.host()
        """
        from .graph_info import DistributeGraphInfo
        return DistributeGraphInfo(None, None, None, ThisHost.host())
