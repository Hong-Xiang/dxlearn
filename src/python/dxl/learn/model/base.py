from doufo import Function, List
from doufo.collections import concat
from abc import abstractmethod
from dxl.learn.config import config_with_name

__all__ = ['Model', 'Stack', 'Residual']


class Model(Function):
    _nargs = None
    _nouts = None

    class KEYS:
        class CONFIG:
            pass

        class MODEL:
            pass

    def __init__(self, name):
        self.config = config_with_name(name)
        self.is_built = False

    def __call__(self, *args):
        if not self.is_built:
            self.build(*args)
            self.is_built = True
        return self.unbox()(*args)

    @property
    @abstractmethod
    def parameters(self):
        pass

    def unbox(self):
        return self.kernel

    @abstractmethod
    def kernel(self, *args):
        pass

    def build(self, *args):
        pass

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


def parameters(ms):
    return concat([m.parameters for m in ms if isinstance(m, Model)])


class Stack(Model):
    _nargs = 1
    _nouts = 1

    def __init__(self, models, name='stack'):
        super().__init__(name)
        self.models = List(models)

    def kernel(self, x):
        for m in self.models:
            x = m(x)
        return x

    @property
    def parameters(self):
        return parameters(self.models)


class Merge(Model):
    _nargs = 1
    _nouts = 1

    def __init__(self, models, merger, name='merger'):
        super().__init__(name)
        self.models = models
        self.merger = merger

    def kernel(self, x):
        xs = [m(x) for m in self.models]
        return self.merger(xs)

    @property
    def parameters(self):
        return parameters(self.models + [self.merger])


class Residual(Model):
    _nargs = 1
    _nouts = 1

    class KEYS(Model.KEYS):
        class CONFIG(Model.KEYS.CONFIG):
            RATIO = 'ratio'

    def __init__(self, name, model, ratio=None):
        super().__init__(name)
        self.model = model
        self.config.update_value_and_default(self.KEYS.CONFIG.RATIO, ratio, 0.3)

    def kernel(self, x):
        return x + self.config[self.KEYS.CONFIG.RATIO] * self.model(x)

    @property
    def parameters(self):
        return parameters([self.model])
