"""
Session creating, managing module.

For most cases, only

 - make_session
 - maks_distribute_session
 - ThisSession

are required.
"""
import tensorflow as tf
from contextlib import contextmanager
from .config import ConfigurableWithName
from .distribute import Server
from abc import ABCMeta, abstractmethod


class SessionBase(ConfigurableWithName, metaclass=ABCMeta):
  _raw_session = None

  class KEYS:
    class CONFIG:
      IS_DEFAULT = 'is_default'
      IS_ALLOW_GROWTH = 'is_allow_growth'
      IS_LOG_DEVICE_PLACEMENT = 'is_log_device_placement'

  def __init__(self,
               name='session',
               is_default=None,
               is_allow_growth=None,
               is_log_device_placement=None):
    ConfigurableWithName.__init__(self, name)
    self.update_config(self.KEYS.CONFIG.IS_DEFAULT, is_default)
    self.update_config(self.KEYS.CONFIG.IS_ALLOW_GROWTH, is_allow_growth)
    self.update_config(self.KEYS.CONFIG.IS_LOG_DEVICE_PLACEMENT,
                       is_log_device_placement)

  def get_session_config(self):
    config = tf.ConfigProto()
    if self.config(self.KEYS.CONFIG.IS_ALLOW_GROWTH) or self.config(
        self.KEYS.CONFIG.IS_ALLOW_GROWTH) is None:
      config.gpu_options.allow_growth = True
    if self.config(self.KEYS.CONFIG.IS_LOG_DEVICE_PLACEMENT):
      config.log_device_placement = True
    return config

  @abstractmethod
  def _create_session(self):
    """
    Return tensorflow session.
    """
    pass

  def _pre_session_creation(self):
    pass

  def _post_session_created(self):
    from .distribute import Cluster, ThisHost
    if Cluster.cluster() is None:
      self.run(tf.global_variables_initializer())
    elif ThisHost.is_master():
      self.run(tf.global_variables_initializer())

  def session(self):
    if self._raw_session is None:
      self._pre_session_creation()
      self._raw_session = self._create_session()
      self._post_session_created()
    return self._raw_session

  @property
  def graph(self):
    return self.session().graph

  def run(self, *args, **kwargs):
    with ThisSession.session_scope(self):
      return ThisSession.session().run(*args, **kwargs)


class Session(SessionBase):
  def __init__(self, name='session'):
    super().__init__(name)

  def _create_session(self):
    return tf.Session(config=self.get_session_config())

  def reset(self):
    tf.reset_default_graph()


class SessionDistribute(SessionBase):
  def __init__(self, name='session', target=None):
    super().__init__(name=name)
    self.target = target

  def _create_session(self):
    return tf.Session(self.target, config=self.get_session_config())


class SessionMonitored(SessionDistribute):
  class KEYS:
    class CONFIG(SessionBase.KEYS.CONFIG):
      CHECKPOINT_DIR = 'checkpoint_dir'

  def __init__(self, name='session', target=None, checkpoint_dir='./save/'):
    super().__init__(name=name, target=target)
    self.checkpoint_dir = checkpoint_dir

  def _create_session(self):
    from .distribute import ThisHost, Master
    master = Master.master_host().job_name
    if ThisHost.is_master():
      creator = tf.train.ChiefSessionCreator(
          master=self.target,
          config=self.get_session_config(),
          checkpoint_dir=self.checkpoint_dir)
    else:
      creator = tf.train.WorkerSessionCreator(
          master=self.target, config=self.get_session_config())
    return tf.train.MonitoredSession(session_creator=creator, )

  def _post_session_created(self):
    pass


class ThisSession:
  _session = None

  @classmethod
  def warp_session(cls):
    return cls._session

  @classmethod
  def reset(cls):
    cls._session.reset()
    cls._session = None

  @classmethod
  def session(cls):
    if cls.warp_session is None:
      return None
    return cls.warp_session().session()

  @classmethod
  def run(cls, *args, **kwargs):
    return cls.session().run(*args, **kwargs)

  @classmethod
  def set_session(cls, session=None):
    if cls._session is not None:
      raise TypeError("Default session is set.")
    if session is None:
      cls._session = tf.get_default_session()
    else:
      cls._session = session
    return cls._session

  @classmethod
  def set_to_default_if_none_is_set(cls, session):
    if cls._session is None:
      cls._session = session
    return cls._session

  @classmethod
  @contextmanager
  def session_scope(cls, session):
    _pre_session = cls._session
    try:
      cls._session = session
      yield
    except Exception as e:
      raise e
    else:
      pass
    cls._session = _pre_session


def make_session(session_name='session'):
  ThisSession.set_session(Session(session_name))
  return ThisSession.session()


def make_distribute_session(session_name='session', target=None):
  if target is None:
    target = Server.server().target
  ThisSession.set_session(SessionMonitored(session_name, target))
  return ThisSession.session()
