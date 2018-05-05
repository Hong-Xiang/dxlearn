import tensorflow as tf
import json
from typing import Dict
import json
from collections import UserDict
import warnings


class JOB_NAME:
    MASTER = 'master'
    WORKER = 'worker'


class Host:
    """
    Object saving host information.
    """

    def __init__(self,
                 job: str,
                 task: int = 0,
                 ip: str = None,
                 port: int = None):
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

    def __str__(self):
        return json.dumps({
            'job': self.job,
            'task_index': self.task_index,
            'ip': self.ip,
            'port': self.port
        })


class MasterHost:
    """
    Helper class to access master host info globally.
    """
    _host = None

    @classmethod
    def set(cls, job: str, task_index: int = None, ip=None, port=None):
        if job is None:
            job = JOB_NAME.MASTER
        if task_index is None:
            task_index = 0
        if cls._host is not None:
            raise TypeError("Maset host is already set.")
        cls._host = Host(job, task_index, ip, port)

    @classmethod
    def reset(cls):
        cls._host = None

    @classmethod
    def set_master(cls, job_name: str = None, task_index: int = None):
        warnings.warn(
            "MasterHost.set_master is going to be deprecated, use Master.set instead.",
            DeprecationWarning)
        set(job_name, task_index)

    @classmethod
    def master_host(cls):
        """
        Alias to master host.
        """
        warnings.warn(
            "MasterHost.master_host() is going to be deprecated, use MasterHost.host() instead.",
            DeprecationWarning)
        return cls.host()

    @classmethod
    def host(cls):
        return cls._host

    @classmethod
    def is_master(cls, host: Host):
        return host == cls.master_host()


Master = MasterHost


class ThisHost:
    _host: Host = None

    @classmethod
    def set(cls, job, task_index=None, ip=None, port=None):
        cls._host = Host(job, task_index, ip, port)
        return cls._host

    @classmethod
    def reset(cls):
        cls._host = None

    @classmethod
    def set_host(cls,
                 job: str,
                 task: int = 0,
                 ip: str = None,
                 port: int = None):
        warnings.warn(
            "ThisHost.set_host is going to be deprecated, use ThisHost.set instead.",
            DeprecationWarning)
        return cls.set(job, task, ip, port)

    @classmethod
    def host(cls):
        return cls._host

    @classmethod
    def is_me(cls, host: Host):
        """
        Return if given host equals ThisHost.host()
        """
        return cls.host() == host

    @classmethod
    def is_master(cls):
        """
        Return if this host is master.
        """
        return Master.is_master(cls.host())


DEFAULT_CLUSTER_CONFIG = {
    JOB_NAME.MASTER: ['localhost:2221'],
    JOB_NAME.WORKER: ['localhost:2333', 'localhost:2334']
}


class ClusterSpec(UserDict):
    class KEYS:
        NB_WORKERS = 'nb_workers'

    def __init__(self, config):
        super().__init__({})
        from pathlib import Path
        if isinstance(config, (str, Path)):
            with open(config, 'r') as fin:
                self.data.update(json.load(fin))
        elif isinstance(config, dict):
            self.data.update(config)
        elif isinstance(config, ClusterSpec):
            self.data.update(config.data)
        else:
            for k, v in config.items():
                self.data[k] = v

    @property
    def nb_workers(self):
        return len(self.data.get(JOB_NAME.WORKER, []))

    @property
    def jobs(self):
        return tuple(self.keys())

    @property
    def master(self):
        return self.data[JOB_NAME.MASTER]

    @property
    def worker(self):
        return self.data[JOB_NAME.WORKER]

    def __str__(self):
        result = {self.KEYS.NB_WORKERS: self.nb_workers}
        result.update(self.data)
        return json.dumps(result)

    def to_tf(self):
        """
        Convert to tensorflow ClusterSpec
        """
        return tf.train.ClusterSpec(self.data)


class Cluster:
    _cluster = None

    class _Cluster:
        def __init__(self, cluster_spec):
            self.spec = ClusterSpec(cluster_spec)
            self._hosts = []
            for job_name, host_spec in self.spec.items():
                if isinstance(host_spec, dict):
                    for i, v in host_spec.items():
                        ip, port = v.split(':')
                        port = int(port)
                        self._hosts.append(Host(job_name, i, ip, port))
                else:
                    for i, h in enumerate(host_spec):
                        ip, port = h.split(':')
                        port = int(port)
                        self._hosts.append(Host(job_name, i, ip, port))
            self._hosts = tuple(self._hosts)

        def hosts(self):
            return self._hosts

        def host(self, job, task_index):
            for h in cls._cluster:
                if h == Host(job, task):
                    return h
            return None

    @classmethod
    def set(cls, config):
        if cls._cluster is not None:
            msg = "Cluster spec already set with spec:\n{}.".format(
                str(cls._cluster.spec))
            raise TypeError(msg)
        cls._cluster = cls._Cluster(config)
        return cls._cluster

    @classmethod
    def set_cluster(cls, config):
        warnings.warn(
            "Cluster.set_cluster is going to be deprecated, use Cluste.set instead",
            DeprecationWarning)
        return cls.set(config)

    @classmethod
    def cluster(cls):
        return cls._cluster

    @classmethod
    def hosts(cls):
        return cls.cluster().hosts()

    @classmethod
    def host(cls, job: str, task: int):
        """
        Get specific host object.
        """
        return cls.cluster().host(job, task)

    @classmethod
    def reset(cls):
        cls._cluster = None


class Server:
    """
    Singloton Server.
    """
    _server = None

    @classmethod
    def set_server(cls, config=None):
        warnings.warn(
            "Server.set_server is going to be deprecated, use Server.set instead.",
            DeprecationWarning)
        return cls.set()

    @classmethod
    def set(cls):
        """
        Construct server for this process. Requires Cluster, ThisHost ready.
        """
        if cls._server is not None:
            raise TypeError("Server is already constructed.")
        if Cluster.cluster() is None:
            raise TypeError("No cluster specification.")
        if ThisHost.host() is None:
            raise TypeError("No ThisHost specification")
        job = ThisHost.host().job
        task_index = ThisHost.host().task
        cluster = Cluster.cluster().spec.to_tf()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        cls._server = tf.train.Server(
            cluster, job_name=job, task_index=task_index, config=config)
        return cls._server

    @classmethod
    def reset(cls):
        """
        Clear server for this process
        """
        cls._server = None

    @classmethod
    def server(cls):
        """
        Returns: tf.train.Server
        """
        return cls._server

    @classmethod
    def target(cls):
        return cls._server.target

    @classmethod
    def join(cls):
        if cls._server is None:
            raise TypeError("Server is not constructed yet.")
        return cls._server.join()


def load_cluster_configs(config=None):
    if config is None:
        return DEFAULT_CLUSTER_CONFIG
    elif isinstance(config, str):
        with open(config, 'r') as fin:
            return json.load(fin)
    else:
        return config


def make_distribute_host(cluster_config: dict,
                         job,
                         task,
                         server_config=None,
                         master_job=None,
                         master_task_index=None):
    """
    Helper function to create current host as an distribute host.
    Create host, i.e. create server, assign ThisHost, etc.
    The following functions will be avaliable after this function called:
    - `ThisHost.host()`
    - `Server.server()`
    """
    warnings.warn(
        "make_distribute_host is going to be deprecated, use make_cluster instead.",
        DeprecationWarning)
    Cluster.set(cluster_config)
    ThisHost.set(job, task)
    Server.set()
    if master_job is not None:
        Master.set_master(master_job, master_task_index)
    return ThisHost.host()


def make_cluster(cluster_spec, job, task_index, master_host=None):
    Cluster.set(cluster_spec)
    ThisHost.set(job, task_index)
    Server.set()
    if master_host is not None:
        Master.set(master_host.job, master_host.task_index)
    return ThisHost.host()


class Barrier:
    def __init__(self, name, signal_hosts, join_hosts, task_lists=()):
        """
        `task_lists`: tasks to be run on signal_hosts, which is a list of task list,
        as shown below:
        [
            [t0_of_signal_host_0, t1_of_signal_host_0,...],
            [t0_of_signal_host_1, t1_of_singal_host_1,...],
            ...
        ]
        
        This barrier can sync *multiple* tasks run on *multiple* signal hosts
        and serve for *multiple* join hosts.

        Each join host has a queue, and required to dequeue # of tokens equal
        to len(taks_lists)

        Each signal host enqueue one token for eqah queue of join hots.
        """
        self.name = name
        self.signal_hosts = signal_hosts
        self.join_hosts = join_hosts
        self.task_lists = task_lists
        self.signal_ops, self.join_ops = self.construct()

    def _make_queues_and_join_ops_for_join_hosts(self):
        # queues is 1:1 to join hosts
        names = [
            "{}_{}".format(self.name, i)
            for i, _ in enumerate(self.join_hosts)
        ]
        queues = [
            tf.FIFOQueue(
                len(self.signal_hosts), tf.bool, [], name=n, shared_name=n)
            for n, _ in zip(names, self.join_hosts)
        ]
        # join op is barried by dequeue nb_single tokens
        join_ops = [q.dequeue_many(len(self.signal_hosts)) for q in queues]
        return join_ops, queues

    def _make_signal_ops(self, queues):
        from .utils import map_data
        # Expand tasks to tf.Tensor or Op
        task_lists = [map_data(tl) for tl in self.task_lists]

        def make_signal_op_for_one_signal_host(task_list):
            with tf.control_dependencies(task_list):
                _ops = [q.enqueue(False) for q in queues]
            with tf.control_dependencies(_ops):
                return tf.no_op()

        signal_ops = [
            make_signal_op_for_one_signal_host(tl) for tl in task_lists
        ]
        return signal_ops

    def construct(self):
        with tf.name_scope(self.name):
            join_ops, queues = self._make_queues_and_join_ops_for_join_hosts()
            signal_ops = self._make_signal_ops(queues)
        return signal_ops, join_ops

    def barrier(self, host):
        """
    Get barrier op for specified host.
    """
        if host in self.signal_hosts:
            return self.signal_ops[self.signal_hosts.index(host)]
        elif host in self.join_hosts:
            return self.join_ops[self.join_hosts.index(host)]
        else:
            return tf.no_op()

    def run(self):
        from .session import ThisSession
        ThisSession.run(self.barrier(ThisHost.host()))
