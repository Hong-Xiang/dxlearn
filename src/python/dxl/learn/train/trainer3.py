import cntk as C
import tensorflow as tf
from dxl.learn.backend import backend, TensorFlowBackend, CNTKBackend

from doufo import singledispatch
from dxl.learn.session import ThisSession


class Trainer:
    def __init__(self, name, model, objective, optimizer):
        self.name = name
        self.objective = objective
        self.optimizer = optimizer
        self.model = model
        self.trainer = None
        self.is_made = False

    def train(self, feeds, session=None):
        if backend(self.objective) is TensorFlowBackend:
            if session is None:
                session = ThisSession
            return session.run(self.trainer, feeds)
        if backend(self.objective) is CNTKBackend:
            return self.trainer.train_minibatch(feeds)

    def make(self):
        if self.is_made:
            return
        minimized = self.optimizer.minimize(self.objective,
                                            parameters(self.model))
        if backend(self.objective) is TensorFlowBackend:
            self.trainer = minimized
        elif backend(self.objective) is CNTKBackend:
            self.trainer = C.Trainer(self.model, self.objective, minimized)
        else:
            raise NotImplementedError(f"Can not create trainer for {type(self.objective)}")


@singledispatch(nargs=1, nouts=1)
def parameters(model):
    return model.parameters


@parameters.register(tf.Tensor)
def _(model):
    return None


class MultiGPUTrainer(Trainer):
    pass


class DistributeTrainer(Trainer):
    pass
