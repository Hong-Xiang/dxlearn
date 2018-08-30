import tensorflow as tf
import json
from typing import Dict
import json
from collections import UserDict
import warnings
from .host import Host
from pathlib import Path


class JOB_NAME:
    MASTER = 'master'
    WORKER = 'worker'
    PARAMETER_SERVER = 'ps'


class ClusterSpec(UserDict):
    def __init__(self, config):
        super().__init__({})
        self._try_load_from_file(config)
        self._try_load_from_dict(config)
        self._try_load_from_cluster_spec(config)
        self._try_load_from_cview(config)

    def _try_load_from_file(self, config):
        if not isinstance(config, (str, Path)):
            return
        with open(config, 'r') as fin:
            self.data.update(json.load(fin))

    def _try_load_from_dict(self, config):
        if not isinstance(config, dict):
            return
        self.data.update(config)

    def _try_load_from_cluster_spec(self, config):
        if not isinstance(config, ClusterSpec):
            return
        self.data.update(config.data)

    def _try_load_from_cview(self, config):
        from dxl.core.config import CView
        if not isinstance(config, CView):
            return
        for k, v in config.items():
            self.data[k] = v

    @property
    def jobs(self):
        return tuple(self.keys())

    def __str__(self):
        return json.dumps(self.data)

    def unbox(self, target_backend=None):
        """
        Convert to tensorflow ClusterSpec
        """
        return tf.train.ClusterSpec(self.data)


class MasterWorkerClusterSpec(ClusterSpec):
    @classmethod
    def make_local_cluster_config(cls, nb_workers):
        return {
            JOB_NAME.MASTER: ['localhost:2222'],
            JOB_NAME.WORKER:
                ['localhost:{}'.format(2333 + i) for i in range(nb_workers)]
        }

    @classmethod
    def make_local_cluster_spec(cls, nb_workers):
        return MasterWorkerClusterSpec(
            MasterWorkerClusterSpec.make_local_cluster_config(nb_workers))

    @property
    def nb_workers(self):
        return len(self.data.get(JOB_NAME.WORKER, []))

    @property
    def master(self):
        return self.data[JOB_NAME.MASTER]

    @property
    def worker(self):
        return self.data[JOB_NAME.WORKER]


def default_master_worker_cluster_config(self, nb_workers=2):
    return {
        JOB_NAME.MASTER: ['localhost:2221'],
        JOB_NAME.WORKER:
            ['localhost:{}'.format(2333 + i) for i in range(nb_workers)]
    }


class Cluster:
    def __init__(self, spec):
        self.spec = spec
        self.hosts = self.make_hosts()

    def make_hosts(self):
        result = {}
        for job_name, host_spec in self.spec.items():
            if isinstance(host_spec, dict):
                hosts = self._parse_dict_jobs(job_name, host_spec)
            else:
                hosts = self._parse_list_jobs(job_name, host_spec)
            result[job_name] = hosts
        return result

    def _parse_dict_jobs(self, job_name, host_spec):
        result = []
        for i, v in host_spec.items():
            ip, port = v.split(':')
            port = int(port)
            result.append(Host(job_name, i, ip, port))
        return result

    def _parse_list_jobs(self, job_name, host_spec):
        result = []
        for i, h in enumerate(host_spec):
            ip, port = h.split(':')
            port = int(port)
            result.append(Host(job_name, i, ip, port))
        return result

    def find(self, job_or_host, task_index=None):
        if isinstance(job_or_host, Host):
            if task_index is not None:
                raise TypeError(
                    "task_index is expected to be None when Host object is provided."
                )
            job = job_or_host.job
            task_index = job_or_host.task_index
        else:
            job = job_or_host
        for h in self.all_hosts():
            if h == Host(job, task_index):
                return h

    def all_hosts(self):
        result = []
        for k, v in self.hosts.items():
            result += v
        return tuple(result)

    def host(self, job, task_index):
        return self.find(job, task_index)


class MasterWorkerCluster(Cluster):
    def master(self):
        return self.hosts[JOB_NAME.MASTER][0]

    def worker(self, task_index):
        return self.hosts[JOB_NAME.WORKER][task_index]

    @property
    def nb_workers(self):
        return self.spec.nb_workers


class DefaultCluster:
    _cluster = None

    @classmethod
    def set(cls, cluster):
        if cls._cluster is not None:
            msg = "Cluster already set with spec:\n{}.".format(
                str(cls._cluster.spec))
            raise TypeError(msg)
        cls._cluster = cluster
        return cls._cluster

    @classmethod
    def cluster(cls):
        return cls._cluster

    @classmethod
    def reset(cls):
        cls._cluster = None


def default_cluster():
    return DefaultCluster.cluster()


class Server:
    """
    Singloton Server.
    """
    _server = None

    @classmethod
    def set(cls, cluster):
        """
        Construct server for this process. Requires Cluster, ThisHost ready.
        """
        from .host import ThisHost
        if cls._server is not None:
            raise TypeError("Server is already constructed.")
        if DefaultCluster.cluster() is None:
            raise TypeError("No cluster specification.")
        if ThisHost.host() is None:
            raise TypeError("No ThisHost specification")
        job = ThisHost.host().job
        task_index = ThisHost.host().task_index

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        cls._server = tf.train.Server(
            cluster.spec.unbox(),
            job_name=job,
            task_index=task_index,
            config=config)
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
            raise TypeError("Server is not set yet.")
        return cls._server.join()


def make_master_worker_cluster(config, job, task_index=0):
    spec = MasterWorkerClusterSpec(config)
    DefaultCluster.set(MasterWorkerCluster(spec))
    from .host import ThisHost, Master
    ThisHost.set(DefaultCluster.cluster().host(job, task_index))
    Master.set(DefaultCluster.cluster().master())
    Server.set(DefaultCluster.cluster())
    return DefaultCluster.cluster()


def make_ps_worker_cluster(cluster_spec, job, task_index):
    pass


def reset_cluster():
    DefaultCluster.reset()
    from .host import Master, ThisHost
    Master.reset()
    ThisHost.reset()
    Server.reset()


__all__ = [
    'JOB_NAME', 'ClusterSpec', 'MasterWorkerClusterSpec', 'Cluster',
    'MasterWorkerCluster', 'DefaultCluster', 'Server',
    'make_master_worker_cluster', 'reset_cluster'
]
