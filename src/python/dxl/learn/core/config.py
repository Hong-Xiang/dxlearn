import json
from typing import Iterable, Dict
from abc import ABCMeta, abstractmethod
from dxl.fs import Path


class DefaultConfig:
    _default_config = dict()

    def add(self, key):
        raise NotImplementedError


class Configurable(metaclass=ABCMeta):
    def __init__(self):
        self._config = dict()

    def _find_config(self, key):
        return self._config.get(key)

    def config(self, key, is_required=False):
        """
        Returns config by key. If no config is found, return None when `is_required` is not True.
        """
        value = self._find_config(key)
        if is_required and value is None:
            raise KeyError("Key {} not found.".format(key))
        return value

    def update_config(self, key, value, force=False):
        if force or value is not None:
            self._config[key] = value


class ConfigurableWithName(Configurable):
    def __init__(self, name: Path, config: Dict[str, 'Config'] = None):
        super().__init__()
        if isinstance(config, Dict): 
            for key, value in config.items():
                if value:
                    self.update_config(key, value)
        self.name = Path(name)


class ConfigurableWithClass(Configurable):
    def __init__(self, cls):
        self.cls = cls


class ConfigurableJoint(Configurable):
    def __init__(self, configs: Iterable[Configurable]):
        self._configurable_configs = configs

    def _find_config(self, key):
        for c in self._configurable_configs:
            if c.config(key) is not None:
                return c.config(key)
        return None
