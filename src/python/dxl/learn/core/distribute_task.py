from .distribute import Host, Master, ThisHost, make_distribute_host, load_cluster_configs
from .session import ThisSession
from .graph_info import DistributeGraphInfo

class DistributeTask:
    class KEYS:
        class STEPS:
            pass
    """
    Helper class of managing distribute run.
    """

    def __init__(self, distribute_configs):
        self.cfg = load_cluster_configs(distribute_configs)
        self.master_graph = None
        self.worker_graphs = []
        self.master_host = None
        self.hosts = []
        self.master_graph_info = None
        self.steps = {}




    def cluster_init(self, job, task, nb_workers=None):
        if nb_workers is None:
            nb_workers = len(self.cfg['worker'])
        make_distribute_host(self.cfg, job, task, None, 'master', 0)
        self.master_host = Master.master_host()
        self.hosts = [Host('worker', i) for i in range(nb_workers)]
        self.master_graph_info = DistributeGraphInfo(None, None, None,
                                                     self.master_host)
        self.worker_graph_infos = [
            DistributeGraphInfo(None, None, None, h) for h in self.hosts
        ]

    def nb_workers(self):
        return len(self.hosts)

    def add_master_graph(self, g):
        self.master_graph = g

    def add_step(self, name, master_op, worker_ops):
        self.steps[name] = {'master': master_op, 'worker': worker_ops}

    def run_step_of_this_host(self, name):
        if ThisHost.is_master():
            ThisSession.run(self.steps[name]['master'])
        else:
            ThisSession.run(self.steps[name]['worker']
                            [ThisHost.host().task_index])

    def add_worker_graph(self, g):
        self.worker_graphs.append(g)

    def add_host(self, h):
        self.hosts.append(h)

    def worker_graph_on(self, host):
        return self.worker_graphs[host.task_index]

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
        from .graph_info import DistributeGraphInfo
        return DistributeGraphInfo(None, None, None, ThisHost.host())
