"""
Trainable Graph
"""
from .graph import Graph
from .session import ThisSession


class SaverSpec:
    pass


class SummaryWritterSpec:
    pass


def create_saver(name, config):
    pass


def create_summary_writter(name, config):
    pass


class Network(Graph):
    def __init__(self, name):
        self.objectives = dict()

    def train(self, feeds=None, name=None):
        pass

    def inference(self, feeds=None, name=None):
        pass

    def evaluate(self, feeds=None, name=None):
        pass

    def save(self):
        pass

    def load(self):
        """
        Restore saved models.
        """