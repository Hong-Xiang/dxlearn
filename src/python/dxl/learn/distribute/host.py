import tensorflow as tf
import json
from typing import Dict
import json
from collections import UserDict
import warnings

__all__ = ['Host', 'Master', 'ThisHost', 'default_host']


class Host:
    """
    Object saving host information.
    """

    def __init__(self,
                 job_or_host: str,
                 task_index: int = None,
                 ip: str = None,
                 port: int = None):
        """
        Parameters:

        - `job_or_host`: A string of host type, like 'master', 'worker', 'ps', etc. Or a `Host` object.
        - `task`: Index of correspoding node given specific cluster.
        - `ip`: ip, optional, if None, __eq__ will return True to host with any ip.
        - `port`: port, optional, if None, __eq__ will return True to host with any port.
        """
        if isinstance(job_or_host, Host):
            if task_index is None and ip is None and port is None:
                job = job_or_host.job
                task_index = job_or_host.task_index
                ip = job_or_host.ip
                port = job_or_host.port
            else:
                raise TypeError(
                    "When job_or_host is Host, all other paramters needs to be None."
                )
        else:
            job = job_or_host
        self.job = job
        if task_index is None:
            task_index = 0
        self.task_index = task_index
        self.ip = ip
        self.port = port

    def device_prefix(self):
        return "/job:{}/task:{}".format(self.job, self.task_index)

    def __eq__(self, h: 'Host'):
        if self.job != h.job or self.task_index != h.task_index:
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


class Master:
    """
    Helper class to access master host info globally.
    """
    _host = None

    @classmethod
    def set(cls,
            job_or_host: str or Host,
            task_index: int = None,
            ip=None,
            port=None):
        if cls._host is not None:
            raise TypeError("Master already set to {}.".format(cls.host()))
        if job_or_host is None:
            job_or_host = JOB_NAME.MASTER
        cls._host = Host(job_or_host, task_index, ip, port)
        return cls._host

    @classmethod
    def reset(cls):
        cls._host = None

    @classmethod
    def host(cls):
        return cls._host

    @classmethod
    def is_master(cls, host: Host):
        if cls.host() is None:
            raise TypeError("MasterHost is not set yet.")
        return host == cls.host()


class ThisHost:
    _host: Host = None

    @classmethod
    def set(cls, job_or_host, task_index=None, ip=None, port=None):
        cls._host = Host(job_or_host, task_index, ip, port)
        return cls._host

    @classmethod
    def reset(cls):
        cls._host = None

    @classmethod
    def host(cls):
        return cls._host

    @classmethod
    def is_this(cls, host: Host):
        """
        Return if given host equals ThisHost.host()
        """
        if cls.host() is None:
            raise TypeError("ThisHost is not set yet.")
        return cls.host() == host

    @classmethod
    def is_master(cls):
        """
        Return if this host is master.
        """
        return Master.is_master(cls.host())


def default_host():
    return ThisHost.host()
