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
        self.task = task
        self.ip = ip
        self.port = port

    def device_prefix(self):
        return "/job:{}/task:{}".formatm(self.job, self.task)

    def __eq__(self, h: 'Host'):
        if self.job != h.job or self.task != h.task:
            return False
        if self.ip is not None and h.ip is not None and self.ip != h.ip:
            return False
        if self.port is not None and h.port is not None and self.port != h.port:
            return False
        return True


class ThisHost:
    _host: Host = None

    @classmethod
    def set_host(cls, job: str, task: int=0, ip: str=None, port: int=None):
        cls._host = Host(job, task, ip, port)
        return cls._host

    @classmethod
    def host(cls):
        return cls._host


class Server:
    _server = None

    @classmethod
    def set_server(cls, config):
        if cls._server is not None:
            raise TypeError("Server is already constructed.")
        if Cluster.cluster() is None:
            raise TypeError("No cluster specification.")
        if ThisHost.host() is None:
            raise TypeError("No ThisHost specification")
        cls._server = tf.train.Server(Cluster.cluster,
                                      ThisHost.host.job,
                                      ThisHost.host.task,
                                      config=config)
        return cls._server

    @classmethod
    def server(cls):
        return cls._server


class Cluster:
    _cluster_spec = None
    _cluster = None
    _server = None
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
