from dxl.learn.core.graph import Graph
from dxl.learn.test import TestCase
from dxl.learn.core.tensor import Tensor
from dxl.learn.core.config import update_config
import unittest
import tensorflow as tf


class TestGraph(TestCase):
    def test_name_of_subgraph(self):
        class TestGraph(Graph):
            def kernel(self):
                self.subgraphs['subg'] = Graph(self.name / 'subg')

        g = TestGraph('g')
        self.assertNameEqual(g.subgraph('subg'), 'g/subg')

    def test_config_of_subgraph(self):
        update_config('g/subg', {'key': 'value'})

        class TestGraph(Graph):
            def kernel(self):
                subg = self.subgraph('subg')

        g = Graph(
            name='g',
            subgraphs={'subg': lambda g: Graph(name=g.name / 'subg')})
        assert g.subgraph('subg').config('key') == 'value'
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
        update_config('g', {'key1': 1})
        g = Graph('g')
        assert g.config('key1') == 1

    def test_info_name(self):
        g = Graph('g')
        assert g.info.name == 'g'

    def test_info_scope(self):
        g = Graph('g')
        with g.info.variable_scope() as scope:
            assert scope.name == 'g'