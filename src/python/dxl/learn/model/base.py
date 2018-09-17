from doufo import Function, List
from doufo.collections.concatenate import concat
from abc import abstractmethod
from dxl.learn.config import config_with_name

__all__ = ['Model', 'parameters']


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
    def kernel(self, x):
        pass

    @abstractmethod
    def build(self, x):
        pass

    def fmap(self, m):
        from dxl.learn.model.stack import Stack
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
    return concat(List([m.parameters for m in ms if isinstance(m, Model)]))
