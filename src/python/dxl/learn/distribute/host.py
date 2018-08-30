import tensorflow as tf
import json
from typing import Optional
import json
from collections import UserDict
import warnings

from dxl.learn.ctx import GlobalContext
from doufo import dataclass, tagfunc

__all__ = ['Host', 'Master', 'ThisHost', 'default_host']


@dataclass
class Host:
    """
    Object saving host information.
    """
    job: str
    task_index: int = 0
    ip: Optional[str] = None
    port: Optional[int] = None


class ThisHost:
    @classmethod
    def set(cls, host):
        GlobalContext.register(Host, host)

    @classmethod
    def reset(cls):
        GlobalContext.reset(Host)

    @classmethod
    def host(cls):
        return GlobalContext.get(Host)

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


@tagfunc()
def device_prefix(host):
    return "/job:{}/task:{}".format(host.job, host.task_index)


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
