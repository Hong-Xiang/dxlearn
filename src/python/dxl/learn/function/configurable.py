from dxl.learn.config import config_with_name
from abc import ABC, abstractmethod


class ConfigurableFunction(ABC):
    def __init__(self, name):
        self.config = config_with_name(name)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
