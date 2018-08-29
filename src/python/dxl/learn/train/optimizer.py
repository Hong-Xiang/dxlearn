from abc import abstractmethod, ABC
from doufo import singledispatch
import tensorflow as tf
import cntk as C

from dxl.learn.backend import backend, TensorFlowBackend, CNTKBackend, NumpyBackend


class Optimizer(ABC):
    class CONFIG:
        LEARNING_RATE = 'learning_rate'

    @abstractmethod
    def minimize(self, objective, parameters):
        pass


class GradientDecentOptimizer(Optimizer):
    def __init__(self, name, learning_rate):
        self.name = name
        self._config = {self.CONFIG.LEARNING_RATE: learning_rate}

    @property
    def config(self):
        return self._config

    def minimize(self, objective, parameters):
        if backend(objective) is TensorFlowBackend:
            return (tf.train.GradientDescentOptimizer(self.config[self.CONFIG.LEARNING_RATE])
                    .minimize(objective,
                              tf.train.get_or_create_global_step(),
                              parameters))
        if backend(objective) is CNTKBackend:
            return C.sgd(parameters,
                         self.config[self.CONFIG.LEARNING_RATE],
                         C.learners.IGNORE)
        raise NotImplementedError(f"GradientDecentOptimizer does not support {type(objective)} yet.")
