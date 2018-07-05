"""
Trainable Graph
"""
from dxl.learn.core import Model
from dxl.learn.core import ThisSession
from dxl.learn.utils import logger
from dxl.learn.core import Tensor
from dxl.learn.network.trainer import Trainer
from .trainer.global_step import GlobalStep


class AbstractNetwork(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            OBJECTIVE = 'objective'
            ACCURACY = 'accuracy'
            INFERENCES = 'inferences'
            EVALUATE = 'evaluate'
            LABEL = 'label'
            TRAINER = 'trainer'
            GLOBAL_STEP = 'global_step'
        
        class GRAPH(Model.KEYS.GRAPH):
            SUMMARY_WRITER = 'summary_writer'
            SAVER = 'saver'

    def __init__(self,
                 info,
                 model,
                 config=None,
                 *,
                 metrics=None,
                 optimizer=None,
                 summary_writer=None,
                 saver=None):
        self.model = self._make_model(model)
        self.metrics = metrics
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.saver = saver
        self.trainers = []
        self.global_step = GlobalStep() 

        super().__init__(info,
                         tensors=self.model.tensors,
                         graphs=self.model.graphs, 
                         config=config)

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

    def train(self, name=None, feeds=None):
        if name is None:
            trainer = self.tensors[self.trainers[0]]
        else:
            trainer = self.tensors.get(name)
        if trainer is None:
            raise ValueError("Nothing to train, please bind first.")
        
        ThisSession.run(trainer, feeds)
        global_step = ThisSession.run(self.global_step.increased())

        self.on_step_end(global_step)

    def on_step_end(self, step):
        if self.summary_writer:
            summary_step = self.summary_writer.summary_step
            if step % summary_step == 0:
                self.summary_writer.write()

        if self.saver:
            save_step = self.saver.save_step
            if step % save_step == 0:
                self.saver.save()

    def load(self, step=None):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
    
    @property
    def trainer(self):
        return self.trainers


class Network(AbstractNetwork):

    def bind(self,
             name='trainer',
             label=None, 
             infer=None,
             *,
             metrics=None,
             optimizer=None,
             summary_writer=None,
             saver=None):
        if metrics is not None:
            self.metrics = metrics
        if optimizer is not None:
            self.optimizer = optimizer
        if summary_writer is not None:
            self.summary_writer = summary_writer
        if saver is not None:
            self.saver = saver

        if label is not None and infer is not None:
            self._init_trainer(name, label, infer)
        
    def _init_trainer(self, name, label, infer):
        KT = self.KEYS.TENSOR
        objective = self._apply_metrics(label, infer)
        self.tensors[name + '_' + KT.OBJECTIVE] = objective
        train_step = self._apply_trainer(name, objective)
        self.get_or_create_tensor(name, train_step)
        self.trainers.append(name)

    def _apply_metrics(self, label, infer):
        _metrics = self.metrics
        if _metrics is None:
            raise ValueError("metrics is None!!!")
        
        return _metrics(label, infer)

    def _apply_trainer(self, name, objective):
        _optimizer = self.optimizer
        if _optimizer is None:
            raise ValueError("optimizer is None!!!")
        
        trainer = Trainer(name, _optimizer, objective)
        trainer.make()
        return trainer.train_step


# class Network(Model):
#     class KEYS(Model.KEYS):
#         class TENSOR(Model.KEYS.TENSOR):
#             TRAIN = 'train'
#             OBJECTIVE = 'objective'
#             ACCURACY = 'accuracy'
#             INFERENCES = 'inferences'
#             EVALUATE = 'evaluate'
#             LABEL = 'label'
#             STEP = 'step'

#         class GRAPH(Model.KEYS.GRAPH):
#             MAIN = 'main'
#             TRAINER = 'trainer'
#             METRICS = 'metrics'
#             SUMMARY_WRITER = 'summary_writer'
#             SAVER = 'saver'

#     def __init__(self,
#                  info='network',
#                  model:Model):
#         """
#         `objectives`: dict of Tensor/tf.tensor or callable. If objectives is a 
#         dict of callables, these callables should have current Network
#         object as the only input and returns a Tensor/tf.tensor object, note one
#         must **NOT** create new variable in this function since it will be called
#         outside managed scope.

#         `trainer` is trainer for self.tensor('objective')
#         `metrics` is a collection of scalar tensor
#         `summaries` is a collection of summaries, typically some scalar summaries of metrics
#         `saver` is saver object for save/load functionality
#         """
#         KS = self.KEYS.GRAPH
#         super().__init__(
#             info,
#             tensors=tensors,
#             graphs=self._parse_input_config(
#                 graphs, {
#                     KS.TRAINER: trainer,
#                     KS.METRICS: metrics,
#                     KS.SUMMARY_WRITER: summaries,
#                     KS.SAVER: saver
#                 }),
#             config=config)

#     @classmethod
#     def _default_config(cls):
#         c = super()._default_config()
#         c.update({})
#         return c


#     def kernel(self, inputs):
#         return {}


#     def post_kernel_in_scope(self, results):
#         KT = self.KEYS.TENSOR
#         objective = self.apply_metrics(results[KT.LABEL],
#                                        results[KT.INFERENCES])
#         self.apply_trainer(objective)

#     def apply_metrics(self, label, infer):
#         KT, KG = self.KEYS.TENSOR, self.KEYS.GRAPH
#         loss = self.graphs[KG.METRICS](label, infer)
#         self.tensors[KT.OBJECTIVE] = loss
#         # self.tensors[KT.ACCURACY] = acc
#         return loss

#     def apply_trainer(self, objective):
#         KT, KG = self.KEYS.TENSOR, self.KEYS.GRAPH
#         self.graphs[KG.TRAINER].make({KT.OBJECTIVE: objective})
#         self.tensors[KT.TRAIN] = self.graphs[KG.TRAINER].train_step
#         self.tensors[KT.STEP] = GlobalStep()

#     def train(self, name=None, feeds=None):
#         """
#         `name`: name of trainer (in subgraph)
#         """
#         if not self.is_made:
#             self.make()
 
#         KT = self.KEYS.TENSOR
#         trainer = self.tensors[self.KEYS.TENSOR.TRAIN]
#         ThisSession.run(trainer, feeds)
#         step = ThisSession.run(self.tensors[KT.STEP].increased())

#         self.on_end_step(step)

#     def inference(self, name=None, feeds=None):
#         t = self.tensors[self.KEYS.TENSOR.INFERENCES]
#         ThisSession.run(t, feeds)

#     def evaluate(self, name=None, feeds=None):
#         t = self.tensors[self.KEYS.TENSOR.EVALUATE]
#         ThisSession.run(t, feeds)

#     def load(self, step=None):
#         """
#         Restore saved models.
#         """
#         self.saver.load(step)

#     def on_end_step(self, step):
#         KG = self.KEYS.GRAPH
#         if self.graphs.get(KG.SUMMARY_WRITER) is not None:
#             summary_step = self.graphs[KG.SUMMARY_WRITER].summary_step
#             if step % summary_step == 0:
#                 self.graphs[KG.SUMMARY_WRITER].write()

#         if self.graphs.get(KG.SAVER) is not None:
#             save_step = self.graphs[KG.SAVER].save_step
#             if step % save_step == 0:
#                 self.graphs[KG.SAVER].save()
