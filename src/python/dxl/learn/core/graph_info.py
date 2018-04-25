from .distribute import Host
import tensorflow as tf
from contextlib import contextmanager
from dxl.fs import Path


class GraphInfo:
  def __init__(self, name=None, variable_scope=None, reuse=None):
    self._name = name
    self.scope = variable_scope
    self.reuse = reuse

  @property
  def name(self):
    if isinstance(self._name, Path):
      return self._name.n
    return self._name

  def set_name(self, name):
    self._name = name

  @contextmanager
  def variable_scope(self, scope=None, reuse=None):
    if scope is None:
      scope = self.scope
    if scope is None:
      yield
    else:
      if isinstance(scope, Path):
        scope = scope.n
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

  def update(self, name=None, variable_scope=None, reuse=None) -> 'GraphInfo':
    return self.from_dict(self.update_to_dict(name, variable_scope, reuse))


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


__all__ = [GraphInfo, DistributeGraphInfo]