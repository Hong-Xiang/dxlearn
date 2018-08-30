import tensorflow as tf
from abc import ABC, abstractmethod
from doufo.tensor import backends
from .learn_extend import assign_add


class AbstractGlobalStep(ABC):
    @abstractmethod
    def incresed(self, n=1):
        pass

    @abstractmethod
    def current(self):
        pass


class TensorflowGlobalStep(AbstractGlobalStep):
    def increased(self, n):
        with tf.variable_scope(f'global_step_increased_{n}'):
            return assign_add(tf.train.get_or_create_global_step(), n)

    def current_step(self):
        return tf.train.get_or_create_global_step()


from doufo import tagfunc


@tagfunc(nargs=0, nouts=1, ndefs=0)
def global_step():
    return global_step[backends.TensorFlowBackend]()


@global_step.register(backends.TensorFlowBackend)
def _():
    return TensorflowGlobalStep()
