from doufo import Function, List
from doufo.collections import concatenate
from abc import abstractmethod


class Model(Function):
    _nargs = None
    _nouts = None

    class KEYS:
        class CONFIG:
            pass

        class MODEL:
            pass

    def __init__(self, name):
        self._config = {}

    def __call__(self, *args):
        return self.unbox()(*args)

    @property
    @abstractmethod
    def parameters(self):
        pass

    def unbox(self):
        return self.kernel

    @abstractmethod
    def kernel(self):
        pass

    @property
    def config(self):
        return self._config

    def fmap(self, m):
        return Stack([m, self])

    @property
    def nargs(self):
        return self._nargs

    @property
    def nouts(self):
        return self._nouts

    @property
    def ndefs(self):
        return 0


class Stack(Model):
    _nargs = 1
    _nouts = 1

    def __init__(self, models, name='stack'):
        super().__init__(name)
        self.models = models

    def kernel(self, x):
        for m in self.models:
            x = m(x)
        return x

    @property
    def parameters(self):
        return concatenate([m.parameters for m in self.models])


class WrappedFunctionModel(Model):
    def __init__(self, f, name='wrapped_model'):
        super().__init__(name)
        self.wrapped = f

    @property
    def parameters(self):
        return []

    def kernel(self, *args):
        return self.wrapped(*args)


def as_model(f):
    return WrappedFunctionModel(f)
