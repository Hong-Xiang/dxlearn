from ..core import GraphInfo
import tensorflow as tf

from contextlib import contextmanager
from .host import Host

__all__ = ['DistributeGraphInfo']


class DistributeGraphInfo(GraphInfo):
    def __init__(self, name, host, variable_scope=None, reuse=None):
        super().__init__(name, variable_scope, reuse)
        self.host = host

    @contextmanager
    def device_scope(self, host=None):
        """
    In most cases, do not use this function directly, use variable_scope instead.
    """
        if host is None:
            host = self.host
        if host is None:
            yield
            return
        if not isinstance(host, Host):
            raise TypeError('Invalid host {}.'.format(host))
        with tf.device(host.device_prefix()):
            yield

    @contextmanager
    def variable_scope(self, scope=None, reuse=None, *, host=None):
        """
    Providing variable scope (with device scope) inferenced from host and name
    information.
    """
        with self.device_scope(host):
            with GraphInfo.variable_scope(self, scope, reuse) as scope:
                yield scope

    @classmethod
    def from_local_info(cls, info, host):
        return cls(info.name, host, info.scope, info.reuse)

    def update_to_dict(self,
                       name=None,
                       host=None,
                       variable_scope=None,
                       reuse=None):
        result = super().update_to_dict(name, variable_scope, reuse)
        if host is None:
            host = self.host
        result.update({'host': host})
        return result

    def update(self, name=None, host=None, variable_scope=None,
               reuse=None) -> 'DistributeGraphInfo':
        return self.from_dict(
            self.update_to_dict(name, host, variable_scope, host))

    def copy_without_name(self):
        return self.from_dict({
            'host': self.host,
            'variable_scope': self.variable_scope,
            'reuse': self.reuse
        })
