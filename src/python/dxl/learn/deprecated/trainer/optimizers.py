from dxl.learn.core import Graph, NotTrainableVariable, NoOp
from dxl.learn.backend import current_backend
import tensorflow as tf


class Optimizer(Graph):
    def minimize(self, objective):
        return NoOp()


class RMSPropOptimizer(Optimizer):
    class KEYS(Graph.KEYS):
        class GRAPH(Graph.KEYS.GRAPH):
            OPTIMIZER = 'optimizer'

        class TENSOR(Graph.KEYS.TENSOR):
            LEARNING_RATE = 'learning_rate'
            DECAY_LEARNING_RATE = 'decay_learning_rate'

        class CONFIG(Graph.KEYS.CONFIG):
            LEARNING_RATE = 'learning_rate'
            DECAY_RATIO = 'decay_ratio'
            OPTIMIZER_NAME = 'optimizer_name'

    def __init__(self,
                 info,
                 *,
                 learning_rate=None,
                 optimizer_name=None,
                 config=None):
        super().__init__(
            info,
            self._parse_input_config(
                config, {
                    self.KEYS.CONFIG.LEARNING_RATE: learning_rate,
                    self.KEYS.CONFIG.OPTIMIZER_NAME: optimizer_name,
                }))

    @classmethod
    def _default_config(cls):
        return {
            cls.KEYS.CONFIG.DECAY_RATIO: 0.1,
        }

    def _get_optimizer(self, name):
        return tf.train.RMSPropOptimizer(
            self.tensors[self.KEYS.TENSOR.LEARNING_RATE].data)

    # TODO: rework to support different optimizer by class, not by name
    def kernel(self, inputs=None):
        KT, KC, KS = self.KEYS.TENSOR, self.KEYS.CONFIG, self.KEYS.GRAPH
        self.tensors[KT.LEARNING_RATE] = NotTrainableVariable(
            self.info.child_tensor(KT.LEARNING_RATE), [],
            current_backend().float32, self.config(KT.LEARNING_RATE))
        self.tensors[
            KT.DECAY_LEARNING_RATE] = self.tensors[KT.LEARNING_RATE].assign(
                self.tensors[KT.LEARNING_RATE] * self.config(KC.DECAY_RATIO))

        self.tensors[KS.OPTIMIZER] = self._get_optimizer(
            self.config(KC.OPTIMIZER_NAME))

    def minimize(self, *args, **kwargs):
        return self.tensors[self.KEYS.GRAPH.OPTIMIZER].minimize(*args, **kwargs)

    @property
    def decay_learning_rate(self):
        return self.tensors[self.KEYS.TENSOR.DECAY_LEARNING_RATE]

    @property
    def learning_rate(self):
        return self.tensors[self.KEYS.TENSOR.LEARNING_RATE]