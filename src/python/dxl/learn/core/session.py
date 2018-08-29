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
from abc import ABCMeta, abstractmethod
import warnings

from ..backend import current_backend


class TestSession:
    def __init__(self, backend_session):
        self.data = backend_session

    def run(self, fetches, feeds=None):
        from .tensor import Tensor
        if isinstance(fetches, (list, tuple)):
            is_tuple = isinstance(fetches, tuple)
            fetches = [current_backend().maybe_unbox(t) for t in fetches]
            if is_tuple:
                fetches = tuple(fetches)
        fetches = current_backend().maybe_unbox(fetches)
        feeds = current_backend().maybe_unbox(feeds)
        if feeds is not None:
            feeds_with_raw_key = {}
            for k in feeds:
                if isinstance(k, Tensor):
                    feeds_with_raw_key[k.data] = feeds[k]
                else:
                    feeds_with_raw_key[k] = feeds[k]
        else:
            feeds_with_raw_key = None
        return self.data.run(fetches, feeds_with_raw_key)
    
    def session(self):
        return self.data


class SessionBase(ConfigurableWithName):

    class KEYS:
        class CONFIG:
            IS_DEFAULT = 'is_default'
            IS_ALLOW_GROWTH = 'is_allow_growth'
            IS_LOG_DEVICE_PLACEMENT = 'is_log_device_placement'
            IS_RUN_VAR_INIT = 'is_run_var_init'

    @classmethod
    def _default_config(cls):
        return {
            cls.KEYS.CONFIG.IS_DEFAULT: True,
            cls.KEYS.CONFIG.IS_ALLOW_GROWTH: True,
            cls.KEYS.CONFIG.IS_LOG_DEVICE_PLACEMENT: False,
            cls.KEYS.CONFIG.IS_RUN_VAR_INIT: True,
        }

    def __init__(self,
                 name='depsession',
                 *,
                 backend_session=None,
                 is_default=None,
                 is_allow_growth=None,
                 is_log_device_placement=None,
                 is_run_var_init=None):
        super().__init__(
            name,
            config={
                self.KEYS.CONFIG.IS_DEFAULT: is_default,
                self.KEYS.CONFIG.IS_ALLOW_GROWTH: is_allow_growth,
                self.KEYS.CONFIG.IS_LOG_DEVICE_PLACEMENT:
                is_log_device_placement,
                self.KEYS.CONFIG.IS_RUN_VAR_INIT: is_run_var_init
            })
        self._raw_session = backend_session

    def get_session_config(self):
        config = tf.ConfigProto()
        if self.config(self.KEYS.CONFIG.IS_ALLOW_GROWTH) or self.config(
                self.KEYS.CONFIG.IS_ALLOW_GROWTH) is None:
            config.gpu_options.allow_growth = True
        if self.config(self.KEYS.CONFIG.IS_LOG_DEVICE_PLACEMENT):
            config.log_device_placement = True
        return config

    def _create_session(self):
        """
        Return tensorflow depsession.
        """
        raise NotImplementedError

    def _pre_session_creation(self):
        pass

    def _post_session_created(self):
        from ..distribute import DefaultCluster, ThisHost
        if self.config(self.KEYS.CONFIG.IS_RUN_VAR_INIT):
            if DefaultCluster.cluster() is None:
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
    def data(self):
        return self._raw_session

    @property
    def graph(self):
        return self.session().graph

    def run_old(self, *args, **kwargs):
        with ThisSession.session_scope(self):
            fetches = args[0]
            if isinstance(fetches, (list, tuple)):
                fetches = [
                    f if isinstance(f, (tf.Tensor, str)) else f.data
                    for f in fetches
                ]
            elif isinstance(fetches, dict):
                fetches = {
                    k: f if isinstance(f, (tf.Tensor, str)) else f.data
                    for k, f in fetches.items()
                }
            elif not isinstance(fetches, (tf.Tensor, str)):
                fetches = fetches.data
            args_new = [fetches, args[1:]]
            return ThisSession.session().run(*args_new, **kwargs)

    def run(self, fetches, feeds=None):
        from .tensor import Tensor
        if isinstance(fetches, (list, tuple)):
            is_tuple = isinstance(fetches, tuple)
            fetches = [current_backend().maybe_unbox(t) for t in fetches]
            if is_tuple:
                fetches = tuple(fetches)
        fetches = current_backend().maybe_unbox(fetches)
        feeds = current_backend().maybe_unbox(feeds)
        if feeds is not None:
            feeds_with_raw_key = {}
            for k in feeds:
                if isinstance(k, Tensor):
                    feeds_with_raw_key[k.data] = feeds[k]
                else:
                    feeds_with_raw_key[k] = feeds[k]
        else:
            feeds_with_raw_key = None
        return self.session().run(fetches, feeds_with_raw_key)


class Session(SessionBase):
    def __init__(self, name='depsession', **kw):
        super().__init__(name, **kw)

    def _create_session(self):
        return tf.Session(config=self.get_session_config())

    def reset(self):
        tf.reset_default_graph()

    def __enter__(self):
        self.session()
        ThisSession.set_session(self)
        return self

    def __exit__(self, type, value, track):
        self._raw_session = None
        ThisSession.reset()

    def init(self):
        ThisSession.run(tf.global_variables_initializer())
        ThisSession.run(tf.local_variables_initializer())


class ThisSession:
    _session = None

    @classmethod
    def warp_session(cls):
        return cls._session

    @classmethod
    def reset(cls):
        # cls._session.reset()
        cls._session = None

    @classmethod
    def session(cls):
        if cls.warp_session is None:
            return None
        return cls.warp_session().session()

    @classmethod
    def run(cls, *args, **kwargs):
        return cls.warp_session().run(*args, **kwargs)

    @classmethod
    def set_session(cls, session=None):
        if cls._session is not None:
            raise TypeError("Default depsession is set.")
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


def default_session():
    return ThisSession


def make_session(session_name='depsession'):
    """
    A quick helper function to make local depsession.
    """
    ThisSession.set_session(Session(session_name))
    return ThisSession.session()
