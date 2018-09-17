from dxl.learn.model import Model, parameters
from doufo.list import List


class Merge(Model):
    _nargs = 1
    _nouts = 1

    def __init__(self, merger=None, models=None, name='DEFAULT_MERGER'):
        super().__init__(name)
        self.models = models
        self.merger = merger

    def kernel(self, x):
        xs = List([m(x) for m in self.models])
        return self.merger(xs)

    def build(self, x):
        pass

    @property
    def parameters(self):
        return parameters(self.models + [self.merger])
