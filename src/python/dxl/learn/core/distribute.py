import tensorflow as tf
from typing import Dict


class Host:
    """
    Object saving host information.
    """

    def __init__(self, job: str, task: int=0, ip: str=None, port: int=None):
        """
        Parameters:

        - `job`: A string of host type, like 'master', 'worker', 'ps', etc.
        - `task`: Index of correspoding node given specific cluster.
        - `ip`: ip, optional, if None, __eq__ will return True to host with any ip.
        - `port`: port, optional, if None, __eq__ will return True to host with any port.
        """
        self.job = job
        self.task = int(task)
        self.ip = ip
        self.port = port

    @property
    def job_name(self):
        return self.job

    @property
    def task_index(self):
        return self.task

    def device_prefix(self):
        return "/job:{}/task:{}".format(self.job, self.task)

    def __eq__(self, h: 'Host'):
        if self.job != h.job or self.task != h.task:
            return False
        if self.ip is not None and h.ip is not None and self.ip != h.ip:
            return False
        if self.port is not None and h.port is not None and self.port != h.port:
            return False
        return True


class Master:
    _master_host = None

    @classmethod
    def set_master(cls, job_name: str=None, task_index: int=None):
        if job_name is None:
            job_name = 'master'
        if task_index is None:
            task_index = 0
        if cls._master_host is not None:
            raise TypeError("Chief is already set.")
        cls._master_host = Host(job_name, task_index)

    @classmethod
    def master_host(cls):
        return cls._master_host

    @classmethod
    def is_master(cls, host: Host):
        return host == cls.master_host()


class ThisHost:
    _host: Host = None

    @classmethod
    def set_host(cls, job: str, task: int=0, ip: str=None, port: int=None):
        cls._host = Host(job, task, ip, port)
        return cls._host

    @classmethod
    def host(cls):
        return cls._host

    @classmethod
    def is_me(cls, host: Host):
        return cls.host() == host

    @classmethod
    def is_master(cls):
        return Master.is_master(cls.host())


class Server:
    _server = None

    @classmethod
    def set_server(cls, config=None):
        if cls._server is not None:
            raise TypeError("Server is already constructed.")
        if Cluster.cluster() is None:
            raise TypeError("No cluster specification.")
        if ThisHost.host() is None:
            raise TypeError("No ThisHost specification")
        job = ThisHost.host().job
        task_index = ThisHost.host().task
        cluster = Cluster.cluster()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        cls._server = tf.train.Server(cluster,
                                      job_name=job,
                                      task_index=task_index,
                                      config=config)
        return cls._server

    @classmethod
    def server(cls):
        return cls._server

    @classmethod
    def join(cls):
        if cls._server is None:
            raise TypeError("Server is not constructed yet.")
        return cls._server.join()


class Cluster:
    _cluster_spec = None
    _cluster = None
    _hosts = None

    @classmethod
    def dumps(cls):
        if cls._cluster_spec is None:
            return ""
        import json
        return json.dumps(cls._cluster_spec, indent=4, separators=(',', ': '),
                          sort_keys=True)

    @classmethod
    def parse_config(cls, config) -> Dict[str, Dict]:
        if isinstance(config, dict):
            return config
        else:
            import json
            return json.loads(config)

    @classmethod
    def set_cluster(cls, config):
        if cls._cluster_spec is not None:
            msg = "Cluster spec already set:\n{}".format(cls.dumps())
            raise TypeError(msg)
        if cls._cluster is not None:
            raise TypeError("Cluster is already constructed.")
        cls._cluster_spec = cls.parse_config(config)
        cls._hosts = []
        for c, ws in cls._cluster_spec.items():
            if isinstance(ws, dict):
                for i, v in ws.items():
                    ip, port = v.split(':')
                    port = int(port)
                    cls._hosts.append(Host(c, i, ip, port))
            else:
                for i, h in enumerate(ws):
                    ip, port = h.split(':')
                    port = int(port)
                    cls._hosts.append(Host(c, i, ip, port))
        cls._hosts = tuple(cls._hosts)
        cls._cluster = tf.train.ClusterSpec(cls._cluster_spec)
        return cls._cluster

    @classmethod
    def cluster(cls):
        return cls._cluster

    @classmethod
    def hosts(cls):
        return tuple(cls._hosts)

    @classmethod
    def host(cls, job: str, task: int):
        for h in cls._cluster:
            if h == Host(job, task):
                return h
        return None


def make_distribute_host(cluster_config, job, task, server_config=None, master_job=None, master_task_index=None):
    Cluster.set_cluster(cluster_config)
    ThisHost.set_host(job, task)
    Server.set_server(server_config)
    if master_job is not None:
        Master.set_master(master_job, master_task_index)
    return ThisHost.host()


class Barrier:
    def __init__(self, name, worker_hosts, tensors):
        self.name = name
        self.worker_hosts = worker_hosts
        self.tensors = tensors
        self.data = self.construct()

    def construct(self):
        queue = tf.FIFOQueue(len(self.worker_hosts), tf.bool, [],
                             name=self.name, shared_name=self.name)
        if ThisHost.host() in worker_hosts:
            return queue.enqueue(False)
        else:
            return queue.dequeue_many(len(self.worker_hosts))
