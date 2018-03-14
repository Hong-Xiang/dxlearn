from .distribute import Host
from contextlib import contextmanager


class GraphInfo:
    def __init__(self, name=None, variable_scope=None, reuse=None):
        self._name = name
        self.scope = scope
        self.reuse = reuse

    @property
    def name(self):
        return self._name

    @classmethod
    def _unify_graph_info(self, graph_info: GraphInfo):
        if isinstance(graph_info, GraphInfo):
            return GraphInfo(**graph_info)
        return graph_info

    @contextmanager
    def variable_scope(self, scope=None, reuse=None):
        if scope is None:
            scope = self.scope
            reuse = self.reuse
        if scope is None:
            yield
        else:
            with tf.variable_scope(scope, reuse=reuse) as scope:
                yield scope


class DistributeGraphInfo:
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
    def variable_scope(self, scope=None, *, host=None):
        with self.device_scope(host):
            with GraphInfo.variable_scope(self, scope) as scope:
                yield scope
