from dxl.learn.model import Model, parameters
from doufo.collections.concatenate import concat
from doufo.list import List

class Merge(Model):
    _nargs = 1
    _nouts = 1

    def __init__(self, name, merger=None, models=None, axis=0):
        super().__init__(name)
        self.models = models
        self.merger = merger
        self.axis = axis

    def kernel(self, x):
        xs = List([m(x) for m in self.models])
        xs = concat(xs, axis=self.axis)
        return self.merger(xs)

    def build(self, x):
        pass

    def config_models(self, models):
        if all(map(lambda a: isinstance(a, Model), models)):
            self.models = models

    def config_merger(self, merger):
        if isinstance(merger, Model):
            self.merger = merger

    @property
    def parameters(self):
        return parameters(self.models + [self.merger])
