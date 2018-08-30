"""
Session creating, managing module.

For most cases, only

 - make_session
 - maks_distribute_session
 - ThisSession

are required.
"""
from abc import ABC, abstractmethod
from contextlib import contextmanager

import tensorflow as tf

from dxl.learn.backend import TensorFlowBackend
from dxl.learn.ctx import GlobalContext

__all__ = ['TensorFlowSession', 'create_session', 'default_session']


class AbstractSession(ABC):
    @abstractmethod
    def run(self, fetches, feeds=None):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def backend(self):
        pass


class TensorFlowSession(AbstractSession):
    def __init__(self, name=None):
        self.name = name
        self.session = self.create_session()
        self.is_init = False

    @property
    def backend(self):
        return TensorFlowBackend

    def graph(self):
        return self.unbox().graph

    def create_session(self):
        return tf.Session(config=self.create_session_config())

    def reset(self):
        self.session.reset()

    def run(self, fetches, feeds=None):
        if not self.is_init:
            self.init()
        return self.session.run(fetches, feeds)

    def create_session_config(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return config

    def __enter__(self):
        ThisSession.set_session(self)
        self.default = self.session.as_default()
        self.default.__enter__()
        return self

    def __exit__(self, type, value, track):
        self.default.__exit__(type, value, track)
        del self.default
        ThisSession.reset()

    def init(self):
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        self.is_init = True


class ThisSession:
    @classmethod
    def session(cls):
        return GlobalContext.get(AbstractSession)

    @classmethod
    def reset(cls):
        GlobalContext.reset(AbstractSession)

    @classmethod
    def run(cls, fetches, feeds=None):
        return cls.session().run(fetches, feeds)

    @classmethod
    def set_session(cls, session=None):
        if session is None:
            session = tf.get_default_session()
        GlobalContext.register(AbstractSession, session)
        return cls.session()

    @classmethod
    def set_to_default_if_none_is_set(cls, session):
        if cls.session() is None:
            cls._session = session
        return cls._session

    @classmethod
    @contextmanager
    def session_scope(cls, session):
        pre = cls.session()
        if pre is session:
            yield
            return
        try:
            GlobalContext.register(AbstractSession, session)
            yield
        except Exception as e:
            raise e
        else:
            pass
        cls.reset()
        cls.set_session(pre)


def create_session(name, backend=None):
    if backend is None:
        backend = TensorFlowBackend
    if backend is TensorFlowBackend:
        return TensorFlowSession(name)


def default_session():
    return ThisSession.session()
