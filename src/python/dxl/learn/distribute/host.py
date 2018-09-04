import tensorflow as tf
import json
from typing import Optional
import json
from collections import UserDict
import warnings

# from dxl.learn.ctx import GlobalContext
from dxl.core.globals import GlobalContext
from doufo import dataclass, tagfunc
from enum import Enum

__all__ = ['Host', 'Master', 'ThisHost', 'default_host']


class JobName(Enum):
    MASTER = 'master'
    WORKER = 'worker'


@dataclass
class Host:
    """
    Object saving host information.
    """
    job: str
    task_index: int = 0
    ip: Optional[str] = None
    port: Optional[int] = None


class ThisHost(GlobalContext):
    @classmethod
    def host(cls):
        return cls.get()

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


class Master(GlobalContext):
    """
    Helper class to access master host info globally.
    """

    @classmethod
    def set(cls,
            job_or_host: str or Host,
            task_index: int = None,
            ip=None,
            port=None):
        if cls.get() is not None:
            raise TypeError("Master already set to {}.".format(cls.host()))
        if job_or_host is None:
            job_or_host = JobName.MASTER
        host = Host(job_or_host, task_index, ip, port)
        return super().set(host)

    @classmethod
    def host(cls):
        return cls.get()

    @classmethod
    def is_master(cls, host: Host):
        if cls.host() is None:
            raise TypeError("MasterHost is not set yet.")
        return host == cls.host()
