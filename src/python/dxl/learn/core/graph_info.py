from .distribute import Host
import tensorflow as tf
from contextlib import contextmanager


class GraphInfo:
    def __init__(self, name=None, variable_scope=None, reuse=None):
        self._name = name
        self.scope = variable_scope
        self.reuse = reuse

    @property
    def name(self):
        return self._name

    @contextmanager
    def variable_scope(self, scope=None, reuse=None):
        if scope is None:
            scope = self.scope
        if scope is None:
            yield
        else:
            with tf.variable_scope(scope, reuse=reuse) as scope:
                yield scope

    @classmethod
    def from_(cls, graph_info: 'GraphInfo', name=None, variable_scope=None, reuse=None, func=None):
        if name is None:
            name = graph_info.name
        if variable_scope is None:
            variable_scope = graph_info.scope
        if reuse is None:
            reuse = graph_info.reuse
        if func is None:
            def func(n, v, r): return cls(n, v, r)
        return func(name, variable_scope, reuse)


class DistributeGraphInfo(GraphInfo):
    def __init__(self, name=None, variable_scope=None, reuse=None, host: Host=None):
        super().__init__(name, variable_scope, reuse)
        self.host = host

    @contextmanager
    def device_scope(self, host=None):
        if host is None:
            host = self.host
        if host is None:
            yield
        else:
            with tf.device(host.device_prefix()):
                yield

    @contextmanager
    def variable_scope(self, scope=None, reuse=None, *, host=None):
        with self.device_scope(host):
            with GraphInfo.variable_scope(self, scope, reuse) as scope:
                yield scope

    @classmethod
    def from_(cls, graph_info: 'DistributeGraphInfo', name=None, variable_scope=None, reuse=None, host=None, func=None):
        from functools import partial
        if host is None:
            host = graph_info.host
        if func is None:
            func = partial(cls, host=host)
        return super().from_(graph_info, name, variable_scope, reuse, func)
