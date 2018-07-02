import json
from typing import Iterable, Dict
from abc import ABCMeta, abstractmethod
from pathlib import Path
import dxl.core.config as dcc


class DefaultConfig:
    _root = None

    @classmethod
    def root(cls):
        if cls._root is None:
            cls.reset()
        return cls._root

    @classmethod
    def reset(cls):
        cls._root = dcc.CNode()

    @classmethod
    def update(cls, key, value):
        cls.root().update(key, value)


def set_global_config(dct):
    DefaultConfig.root().update([], dct)


def clear_config():
    DefaultConfig.reset()


def update_config(key, value):
    DefaultConfig.root().update(key, value)


class _Config:
    def __init__(self, node, view):
        self._cnode = node
        self._cview = view

    def __call__(self, key: str, value=None):
        """
        Returns config by key. If no config is None, return value.
        """
        return self._cview.get(key, value)

    def update(self, key, value):
        self._cnode.update(key, value)


class Configurable:
    def _create_config(self, cnode):
        raise NotImplementedError

    @classmethod
    def _default_config(cls):
        return {}

    def __init__(self, config=None):
        if config is None:
            config = {}
        config = {k: v for k, v in config.items() if v is not None}
        self.config = self._create_config(config)
        for k, v in self._default_config().items():
            if self.config(k) is None:
                self.config.update(k, v)

    @classmethod
    def _parse_input_config(cls, config, input_config):
        if config is None:
            config = {}
        config = dict(config)
        for k, v in input_config.items():
            if v is not None:
                config[k] = v
        return config

    def update(self, dct):
        for k, v in dct.items():
            if v is not None and v is not ...:
                self.config.update(k, v)


class ConfigurableWithName(Configurable):
    def __init__(self, name: Path, config=None):
        """
        `config`: Dict[str, 'Config'] | dict | None
        """
        self.name = Path(name)
        super().__init__(config)

    def _create_config(self, config):
        DefaultConfig.root().update(str(self.name), config)
        node = DefaultConfig.root().read(str(self.name))
        view = dcc.create_view(DefaultConfig.root(), str(self.name))
        return _Config(node, view)


class ConfigurableWithClass(Configurable):
    def __init__(self, cls, config=None):
        self.cls = cls
        super().__init__()

    def _create_config(self, cnode):
        DefaultConfig.root().update(str(self.cls), config)
        node = DefaultConfig.root().get(str(self.cls))
        view = dcc.create_view(DefaultConfig.root(), str(self.cls))
        return _Config(node, view)


# class ConfigurableJoint(Configurable):
#     def __init__(self, configs: Iterable[Configurable]):
#         self._configurable_configs = configs

#     def _find_config(self, key):
#         for c in self._configurable_configs:
#             if c.config(key) is not None:
#                 return c.config(key)
#         return None
