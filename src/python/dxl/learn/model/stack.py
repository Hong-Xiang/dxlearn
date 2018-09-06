from doufo.list import List
from dxl.learn.model.base import Model, parameters


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

    def build(self, x):
        pass
