from ...core import make_distribute_host, DistributeGraphInfo, Host, Master


def cluster_configs_generator(master_ip, workers_ips, nb_process_per_worker):
  workers = []
  for worker in WORKERS_IO_SUFFIX:
    for p in range(2333, 2333 + NB_PROCESS_PER_WORKER):
      workers.append("192.168.1.{}:{}".format(worker, p))
  return {
      "master": ["192.168.1.{}:2221".format(MASTER_IP_SUFFIX)],
      "worker": workers
  }


sample_cluster_config = {
    "master": ["localhost:2221"],
    "worker": ["localhost:2333", "localhost:2334"]
}


def load_cluster_configs(config=None):
  if config is None:
    return sample_cluster_config
  elif isinstance(config, str):
    with open(config, 'r') as fin:
      return json.load(fin)
  else:
    return config


class DistributeTask:
  """
  Helper class of managing distribute run.
  """

  def __init__(self, distribute_configs):
    self.cfg = distribute_configs
    self.master_graph = None
    self.worker_graphs = []
    self.master_host = None
    self.hosts = []
    self.master_graph_info = None

  def cluster_init(self, job, task, nb_workers=None):
    if nb_workers is None:
      nb_workers = len(self.cfg['worker'])
    from ...core import DistributeGraphInfo
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
      for g in local_graphs:
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


from .utils import DataInfo


