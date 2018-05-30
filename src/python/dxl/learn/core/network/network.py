"""
Trainable Graph
"""
from ..model import Model
from ..session import ThisSession
from ..utils import logger


class Network(Model):
    """
    A network is a trainable Graph.
    A member maybe added.
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            TRAINERS = 'trainers'
            OBJECTIVES = 'objectives'
            SUMMARIES = 'summaries'
            INFERNECES = 'inferences'
            EVALUATE = 'evaluate'

        class SUBGRAPH(Model.KEYS.SUBGRAPH):
            TRAINER = 'trainer'

    def __init__(self,
                 name='network',
                 inputs=None,
                 subgraphs=None,
                 config=None,
                 info=None,
                 *,
                 objectives=None,
                 trainers=None,
                 summaries=None,
                 saver=None,
                 add_trainers=None,
                 add_saver=None):
        """
        `objectives`: dict of Tensor/tf.tensor or callable. If objectives is a 
        dict of callables, these callables should have current Network
        object as the only input and returns a Tensor/tf.tensor object, note one
        must **NOT** create new variable in this function since it will be called
        outside managed scope.

        """
        super().__init__(name, inputs, submodels, info, config)

    @classmethod
    def default_config(cls):
        c = super().default_config()
        c.update({})
        return c

    def _fetech_tensor_maybe_in_dict(self, group_name, name=None):
        if isinstance(self.tensor(group_name), dict) and name is not None:
            return self.tensor(group_name)[name]
        if name is None:
            return self.tensor(group_name)
        raise ValueError("Invalid feteching tensor: {}/{}.".format(
            group_name, name))

    def train(self, name=None, feeds=None):
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
        pass

    def load(self, step=None):
        """
        Restore saved models.
        """
        pass
