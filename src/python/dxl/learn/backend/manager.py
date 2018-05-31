from contextlib import contextmanager


def get_env_default_backend():
    from ._tensorflow import TensorFlow
    return TensorFlow()


class DefaultBackendManager:
    _default_backend = get_env_default_backend()

    @classmethod
    def set_as(cls, backend):
        cls._default_backend = backend

    @classmethod
    def reset(cls):
        cls._default_backend = get_env_default_backend()

    @classmethod
    def backend(cls):
        return cls._default_backend

    @classmethod
    @contextmanager
    def default_backend_as(cls, backend):
        _pre = cls.backend()
        cls.set_as(backend)
        yield
        cls.set_as(_pre)


def current_backend():
    return DefaultBackendManager.backend()