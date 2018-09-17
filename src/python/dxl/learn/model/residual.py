from dxl.learn.model.base import Model
from dxl.learn.model import parameters


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

    def build(self, x):
        pass
