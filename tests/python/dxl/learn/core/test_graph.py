from dxl.learn.core.graph import Graph
from dxl.learn.core.tensor import Tensor
import unittest
import tensorflow as tf


def add_default_config(c):
    pass


class TestGraph(unittest.TestCase):
    def test_config(self):
        with tf.Graph().as_default():
            g = Graph('g', config={'some_key': 1})
            assert g.config('some_key') == 1

    def test_config_of_subgraph(self):
        class TestGraph(Graph):
            def kernel(self):
                subg = self.subgraph('subg')

        g = Graph(
            name='g',
            subgraphs={'subg': lambda g: Graph(name=g.name / 'subg')},
            config={'subg': {
                'some_key': 1
            }})
        assert g.subgraph('subg').config('some_key') == 1
        assert g.subgraph('subg').info.name == 'g/subg'

    def test_access_tensor(self):
        class TestGraph(Graph):
            def kernel(self):
                self.tensors['x'] = Tensor(
                    tf.constant('x'),
                    None,
                    self.info.update(name=self.name / 'x'),
                )

        g = TestGraph('test_g')
        assert isinstance(g.tensor('x'), Tensor)
        assert str(g.tensor('x').info.name) == 'test_g/x'

    def test_access_config(self):
        add_default_config({'g': {'key1': 1}})
        g = Graph('g')
        assert g.config('key1') == 1

    def test_info_name(self):
        g = Graph('g')
        assert g.info.name == 'g'

    def test_info_scope(self):
        g = Graph('g')
        with g.info.variable_scope() as scope:
            assert scope.name == 'g'