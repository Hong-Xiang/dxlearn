from doufo import singledispatch
import tensorflow as tf
import numpy as np
from doufo.tensor.backends import *

__all__ = ['backend', 'TensorFlowBackend', 'CNTKBackend', 'NumpyBackend']


class Backend:
    pass


class TensorFlowBackend(Backend):
    pass


class CNTKBackend(Backend):
    pass


class NumpyBackend(Backend):
    pass


@backend.register(tf.Tensor)
def _(b):
    return TensorFlowBackend


try:
    import cntk as C
except:
    pass
else:
    @backend.register(C.Variable)
    @backend.register(C.Function)
    def _(b):
        return CNTKBackend


    @singledispatch(nargs=1, nouts=1)
    def backend(b):
        backend_map = {
            tf: TensorFlowBackend,
            C: CNTKBackend,
            np: NumpyBackend
        }
        if b in backend_map:
            return backend_map[b]
        raise TypeError(f"Can't specify backend for {type(b)}")


@backend.register(np.ndarray)
def _(b):
    return NumpyBackend
