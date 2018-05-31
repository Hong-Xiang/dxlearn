import tensorflow as tf
import json
from typing import Dict
import json
from collections import UserDict
import warnings

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
