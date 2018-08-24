from doufo import Function, List
from doufo.collections import concatenate
from abc import abstractmethod


class Model(Function):
    _nargs = None
    _nouts = None


    def __init__(self, name):
        self._config = {}
        self.is_built = False
        self.kernel = None
        self.models = []

    def __call__(self, *args):
        return self.unbox(*args)

    @property
    def parameters(self):
        return concatenate([m.parameters for m in self.models])

    @abstractmethod
    def build(self, *args):
        pass

    def unbox(self):
        return self.kernel

    @property
    def config(self):
        return self._config

    @property
    def nargs(self):
        return self._nargs

    @property
    def nouts(self):
        return self._nouts


class ModelNeedBuild(Model):
    def __call__(self, *args, **kwargs):
        if not self.is_built:
            self.kernel = self.build(*args)
            self.is_built = True
        return self.unbox(*args)
