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
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            TRAIN = 'train'
            OBJECTIVES = 'objectives'
            METRICS = 'metrics'
            INFERNECES = 'inferences'
            EVALUATE = 'evaluate'

        class SUBGRAPH(Model.KEYS.SUBGRAPH):
            TRAINER = 'trainer'
            SUMMARY_WRITER = 'summary_writer'
            SAVER = 'saver'

    def __init__(self,
                 info='network',
                 inputs=None,
                 subgraphs=None,
                 config=None,
                 *,
                 trainer=None,
                 metrics=None,
                 summaries=None,
                 saver=None,
                 is_add_trainer=None,
                 is_add_saver=None):
        """
        `objectives`: dict of Tensor/tf.tensor or callable. If objectives is a 
        dict of callables, these callables should have current Network
        object as the only input and returns a Tensor/tf.tensor object, note one
        must **NOT** create new variable in this function since it will be called
        outside managed scope.

        """
        super().__init__(name, inputs, submodels, info, config)

    @classmethod
    def _default_config(cls):
        c = super()._default_config()
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

    def train_multiple_steps(self, nb_steps=None, name=None, feeds=None):
        pass

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
