class DistributeGraphInfo(GraphInfo):
    def __init__(self,
                 name=None,
                 variable_scope=None,
                 reuse=None,
                 host: Host = None):
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
        else:
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
    def from_graph_info(cls,
                        distribute_graph_info: 'DistributeGraphInfo',
                        name=None,
                        variable_scope=None,
                        reuse=None,
                        host=None):
        return cls.from_dict(
            distribute_graph_info.update_to_dict(name, variable_scope, reuse,
                                                 host))

    def update_to_dict(self,
                       name=None,
                       variable_scope=None,
                       reuse=None,
                       host=None):
        result = super().update_to_dict(name, variable_scope, reuse)
        if host is None:
            host = self.host
        result.update({'host': host})
        return result

    def update(self, name=None, variable_scope=None, reuse=None,
               host=None) -> 'GraphInfo':
        return self.from_dict(
            self.update_to_dict(name, variable_scope, reuse, host))

    def copy_without_name(self):
        return self.from_dict({
            'variable_scope': self.variable_scope,
            'reuse': self.reuse,
            'host': self.host
        })
