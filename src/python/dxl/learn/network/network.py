"""
Trainable Graph
"""
from dxl.learn.core import Model
from dxl.learn.core import ThisSession
from dxl.learn.utils import logger
from dxl.learn.core import Tensor
from dxl.learn.core.global_ctx import get_global_context
from dxl.learn.network.trainer import Trainer
from dxl.learn.network.summary import SummaryWriter
from .trainer.global_step import GlobalStep


class AbstractNetwork(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            OBJECTIVE = 'objective_'
            ACCURACY = 'accuracy_'
            INFERENCES = 'inferences_'
            EVALUATE = 'evaluate_'
            LABEL = 'label_'
            TRAINER = 'trainer_'
            GLOBAL_STEP = 'global_step'

        class GRAPH(Model.KEYS.GRAPH):
            LOSS = 'loss_'
            OPTIMIZER = 'optimizer_'
            SUMMARY_WRITER = 'summary_writer_'
            SAVER = 'saver_'

    def __init__(self,
                 info,
                 model,
                 config=None):
        self.model = self._make_model(model)

        self.trainers = []
        self.global_step = GlobalStep()

        super().__init__(info, config=config)

    @classmethod
    def _default_config(cls):
        c = super()._default_config()
        c.update({})
        return c

    def _make_model(self, model):
        # make model
        if not model.is_made:
            model()
        return model

    def _add_tensor(self, name, t):
        if name in self.tensors:
            raise ValueError("{} is already exits".format(name))
        self.tensors[name] = t

    def _update_bind(self, binds):
        for k, v in binds:
            if v is not None:
                if k in self.graphs:
                    raise ValueError("{} is already exits".format(k))
                self.graphs[k] = v

    def apply_loss(self, name, label, infer):
        _loss = self.graphs.get(self.KEYS.GRAPH.LOSS + name)
        if _loss is None:
            raise ValueError("loss is None, please bind first!")
        return _loss(label, infer)

    def apply_optimizer(self, name):
        _optimizer = self.graphs.get(self.KEYS.GRAPH.OPTIMIZER + name)
        if _optimizer is None:
            raise ValueError("optimizer is None, please bind first!")
        return _optimizer

    def apply_trainer(self, name, objective):
        if name in self.tensors:
            raise ValueError("trainer {} is already exits".format(name))
        _optimizer = self.apply_optimizer(name)
        trainer = Trainer(name, _optimizer, objective)
        trainer.make()
        return trainer.train_step

    def get_objective(self, name=None):
        if name is None:
            name = 'default'
        return self.tensors.get(self.KEYS.TENSOR.OBJECTIVE+name)

    def train(self, name=None, feeds=None):
        if name is None:
            name = 'default'
        trainer = self.tensors.get(name)
        if trainer is None:
            raise ValueError("Nothing to train, please bind first.")

        ThisSession.run(trainer, feeds)
        global_step = ThisSession.run(self.global_step.increased())

        self.on_step_end(name, global_step)

    def evaluate(self, name=None, feeds=None):
        with get_global_context().test_phase():
            return ThisSession.run(self.tensors.get(name), feeds)

    def on_step_end(self, name, step):
        KG = self.KEYS.GRAPH
        summary_writer = self.graphs.get(KG.SUMMARY_WRITER + name)
        if summary_writer is not None:
            summary_writer.auto_run()

        saver = self.graphs.get(KG.SAVER + name)
        if saver is not None:
            saver.auto_save()

    def load(self, step=None):
        raise NotImplementedError

    @property
    def traner(self):
        return self.trainers


class Network(AbstractNetwork):

    def bind(self,
             name='default',
             loss=None,
             optimizer=None,
             summary_writer=None,
             saver=None):
        KG = self.KEYS.GRAPH
        _names = (
            KG.LOSS + name,
            KG.OPTIMIZER + name,
            KG.SUMMARY_WRITER + name,
            KG.SAVER + name
        )
        _binds = (loss, optimizer, summary_writer, saver)
        self._update_bind(zip(_names, _binds))

    def build_trainer(self, label, infer, name='default'):
        KT = self.KEYS.TENSOR
        objective = self.apply_loss(name, label, infer)
        self._add_tensor(KT.OBJECTIVE + name, objective)

        train_step = self.apply_trainer(name, objective)
        self._add_tensor(name, train_step)
        self.trainers.append(name)
