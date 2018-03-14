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

    def __init__(self, name='session', is_default=None, is_allow_growth=None, is_log_device_placement=None):
        ConfigurableWithName.__init__(name)
        self.update_config(self.KEYS.CONFIG.IS_DEFAULT, is_default)
        self.update_config(self.KEYS.CONFIG.IS_ALLOW_GROWTH, is_allow_growth)
        self.update_config(self.KEYS.CONFIG.IS_LOG_DEVICE_PLACEMENT,
                           is_log_device_placement)

    def get_session_config(self):
        config = tf.ConfigProto()
        if self.config(self.KEYS.CONFIG.ALLOW_GROWTH):
            config.gpu_options.allow_growth = True
        if self.config(self.KEYS.CONFIG.ALLOW_GROWTH):
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
        self.run(tf.global_variables_initializer())

    def session(self):
        if _raw_session is None:
            self._pre_session_creation()
            self._raw_session = self._create_session()
            self._post_session_created()
        return self._raw_session

    @property
    def graph(self):
        return self.session().graph

    def run(self, *args, **kwargs):
        with ThisSession.session_scope(self):
            ThisSession.session().run(*args, **kwargs)


class Session(SessionBase):
    def __init__(self, name='session'):
        super().__init__(name)

    def _create_session(self):
        return tf.Session(config=self.get_session_config())


class SessionDistributed(SessionBase):
    def __init__(self, name='session', target=None):
        super().__init__(name=name)
        self.target = target

    def _create_session(self):
        return tf.Session(self.target, config=self.get_session_config())


class SessionMonitored(Session):
    class KEYS:
        class CONFIG(SessionBase.KEYS.CONFIG):
            CHECKPOINT_DIR = 'checkpoint_dir'

    def __init__(self, name='session', target=None,
                 checkpoint_dir='./save/', **kw):
        self._target = target
        super().__init__(name=name, checkpoint_dir=checkpoint_dir, **kw)

    def run(self, *args, **kwargs):
        return self.nodes[NodeKeys.MAIN].run(*args, **kwargs)

    def _create_session(self):
        sess = tf.train.MonitoredTrainingSession(config=self._get_session_config(),
                                                 checkpoint_dir=self.config('checkpoint_dir'))
        self.register_main_node(sess)


class ThisSession:
    _session = None

    @classmethod
    def warp_session(self):
        return cls._session

    @classmethod
    def session(cls):
        if cls.warp_session is None:
            return None
        return cls.warp_session.session()

    @classmethod
    def set_session(cls, session=None):
        if cls._session is not None:
            raise TypeError("Default session is set.")
        if session is None:
            return tf.get_default_session()
        return cls._session

    @classmethod
    def set_to_default_if_none_is_set(cls, session):
        if cls._session is None:
            cls._session = session
        return cls._session

    @classmethod
    @contextmanager
    def session_scope(cls, session):
        _pre_session = _session
        try:
            cls._session = session
            yield
        except Exception as e:
            raise e
        else:
            pass
        _session = _pre_session
