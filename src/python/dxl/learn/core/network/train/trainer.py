from ...graph import Graph
from ...tensor import VariableV2 as Variable
from dxl.learn.backend import current_backend
import tensorflow as tf


class Trainer(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            OBJECTIVE = 'objective'
            TRAIN_STEP = 'train_step'
            LEARNING_RATE = 'learning_rate'

    def __init__(self, info, objective, config, *, optimizer):
        super().__init__(
            self,
            name=info.name,
            inputs={self.KEYS.TENSOR.OBJECTIVE: objective},
            config=config)

    def kernel(self):
        self.tensors[self.KEYS.TENSOR.LEARNING_RATE] = Variable(
            self.info.child_scope(self.KEYS.TENSOR.LEARNING_RATE), [],
            current_backend().float32,
            self.config(self.KEYS.TENSOR.LEARNING_RATE))

        if self.optimizer is None:
            self.optimizer = current_backend.get_optimizer(
                self.config('optimizer'), learning_rate=self.learning_rate)
        self.tensors[self.KEYS.TENSOR.TRAIN_STEP] = self.optimizer.minimize(
            self.objective)
        self.tensors[self.KEYS.TENSOR.MAIN] = self.train_step

    @property
    def learning_rate(self):
        return self.tensor(self.KEYS.TENSOR.LEARNING_RATE)

    @property
    def objective(self):
        return self.tensor(self.KEYS.TENSOR.OBJECTIVE)

    @property
    def train_step(self):
        return self.tensor(self.KEYS.TENSOR.TRAIN_STEP)