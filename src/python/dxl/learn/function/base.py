from doufo import Function


class ConfigurableFunction(Function):
    _nargs = None
    _nouts = None

    def __init__(self, name):
        self.name = name
        self._config = {}

    def __call__(self, *args):
        return self.unbox()(*args)

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
