import tensorflow as tf
from contextlib import contextmanager
from pathlib import Path
import pathlib
import json


class GraphInfo:
    def __init__(self, name=None, variable_scope=None, reuse=None):
        if name is not None:
            self.name = Path(name)
        else:
            self.name = name
        if isinstance(variable_scope, Path):
            variable_scope = str(variable_scope)
        self.scope = variable_scope
        if self.scope is None:
            self.scope = str(self.name)
        self.reuse = reuse

    def __str__(self):
        return json.dumps({
            'name': str(self.name),
            'scope': str(self.scope),
            'reuse': self.reuse
        })

    def relative_scope(self, scope):
        if isinstance(scope, Path):
            scope = scope.name
            current_scope = tf.get_variable_scope().name
            if scope.startswith(current_scope):
                scope = str(Path(scope).relative_to(scope))
                if scope == '.':
                    scope = ''
        return scope

    def relative_name(self):
        return self.relative_scope(self.name)

    @contextmanager
    def variable_scope(self, scope=None, reuse=None):
        if scope is None and self.scope is None:
            yield
            return
        if scope is None:
            scope = self.scope
        scope = self.relative_scope(scope)
        if scope is tf.get_variable_scope():
            yield scope
        else:
            with tf.variable_scope(scope, reuse=reuse) as scope:
                self.scope = scope
                yield scope

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    @classmethod
    def from_graph_info(cls,
                        graph_info: 'GraphInfo',
                        name=None,
                        variable_scope=None,
                        reuse=None):
        return cls.from_dict(
            graph_info.update_to_dict(name, variable_scope, reuse))

    def update_to_dict(self, name=None, variable_scope=None, reuse=None):
        if name is None:
            name = self.name
        if variable_scope is None:
            variable_scope = self.scope
        if reuse is None:
            reuse = self.reuse
        return {'name': name, 'variable_scope': variable_scope, 'reuse': reuse}

    def update(self, name=None, variable_scope=None,
               reuse=None) -> 'GraphInfo':
        return self.from_dict(self.update_to_dict(name, variable_scope, reuse))

    def child_scope(self, name):
        cname = Path(self.name) / name
        if not isinstance(self.scope, (str, Path)):
            with self.variable_scope():
                with tf.variable_scope(name) as scope:
                    pass
        else:
            scope = str(cname)
        return self.update(cname, variable_scope=scope)

    def child_tensor(self, name):
        cname = Path(self.name) / name
        return self.update(cname, variable_scope=self.scope)

    def copy_without_name(self):
        return self.from_dict({
            'variable_scope': self.variable_scope,
            'reuse': self.reuse
        })

    def erase_name(self):
        return self.copy_without_name()


# class DistributeGraphInfo(GraphInfo):
#     def __init__(self,
#                  name=None,
#                  variable_scope=None,
#                  reuse=None,
#                  host: Host = None):
#         super().__init__(name, variable_scope, reuse)
#         self.host = host

#     @contextmanager
#     def device_scope(self, host=None):
#         """
#     In most cases, do not use this function directly, use variable_scope instead.
#     """
#         if host is None:
#             host = self.host
#         if host is None:
#             yield
#         else:
#             with tf.device(host.device_prefix()):
#                 yield

#     @contextmanager
#     def variable_scope(self, scope=None, reuse=None, *, host=None):
#         """
#     Providing variable scope (with device scope) inferenced from host and name
#     information.
#     """
#         with self.device_scope(host):
#             with GraphInfo.variable_scope(self, scope, reuse) as scope:
#                 yield scope

#     @classmethod
#     def from_graph_info(cls,
#                         distribute_graph_info: 'DistributeGraphInfo',
#                         name=None,
#                         variable_scope=None,
#                         reuse=None,
#                         host=None):
#         return cls.from_dict(
#             distribute_graph_info.update_to_dict(name, variable_scope, reuse,
#                                                  host))

#     def update_to_dict(self,
#                        name=None,
#                        variable_scope=None,
#                        reuse=None,
#                        host=None):
#         result = super().update_to_dict(name, variable_scope, reuse)
#         if host is None:
#             host = self.host
#         result.update({'host': host})
#         return result

#     def update(self, name=None, variable_scope=None, reuse=None,
#                host=None) -> 'GraphInfo':
#         return self.from_dict(
#             self.update_to_dict(name, variable_scope, reuse, host))

#     def copy_without_name(self):
#         return self.from_dict({
#             'variable_scope': self.variable_scope,
#             'reuse': self.reuse,
#             'host': self.host
#         })

__all__ = ['GraphInfo']