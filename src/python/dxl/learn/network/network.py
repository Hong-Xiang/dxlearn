"""
Trainable Graph
"""
from ..model import Model
from ..session import ThisSession
from ...utils import logger


class Network(Model):
    """
    A network is a trainable Graph.
    A member maybe added.
    A Network is restricted to has at most one objective/trainer/train_step.
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            TRAIN = 'train'
            OBJECTIVE = 'objective'
            METRICS = 'metrics'
            INFERNECES = 'inferences'
            EVALUATE = 'evaluate'

        class SUBGRAPH(Model.KEYS.SUBGRAPH):
            TRAINER = 'trainer'
            SUMMARY_WRITER = 'summary_writer'
            SAVER = 'saver'

    def __init__(self,
                 info='network',
                 *,
                 tensors=None,
                 graphs=None,
                 config=None,
                 trainer=None,
                 metrics=None,
                 summaries=None,
                 saver=None):
        """
        `objectives`: dict of Tensor/tf.tensor or callable. If objectives is a 
        dict of callables, these callables should have current Network
        object as the only input and returns a Tensor/tf.tensor object, note one
        must **NOT** create new variable in this function since it will be called
        outside managed scope.

        """
        KS = self.KEYS.SUBGRAPH
        super().__init__(
            info,
            tensors=tensors,
            graphs=self._parse_input_config(graphs, {KS.TRAINER: trainer}),
            config=config)

    @classmethod
    def _default_config(cls):
        c = super()._default_config()
        c.update({})
        return c

    def _fetech_tensor_maybe_in_dict(self, group_name, name=None):
        if isinstance(self.tensors[group_name], dict) and name is not None:
            return self.tensors[group_name][name]
        if name is None:
            return self.tensors[group_name]
        raise ValueError("Invalid feteching tensor: {}/{}.".format(
            group_name, name))

    def train(self, name=None, feeds=None):
        """
        `name`: name of trainer (in subgraph)
        """
        trainer = self._fetech_tensor_maybe_in_dict(self.KEYS.TENSOR.TRAINERS,
                                                    name)
        trainer.train(feeds)

    def inference(self, name=None, feeds=None):
        t = self._fetech_tensor_maybe_in_dict(self.KEYS.TENSOR.INFERNECES,
                                              name)
        ThisSession.run(t, feeds)

    def evaluate(self, name=None, feeds=None):
        t = self._fetech_tensor_maybe_in_dict(self.KEYS.TENSOR.EVALUATE, name)
        ThisSession.run(t, feeds)

    def save(self):
        self.saver.save()

    def load(self, step=None):
        """
        Restore saved models.
        """
        self.saver.load(step)
