import unittest
from pathlib import Path

import pytest
import tensorflow as tf

from dxl.learn.core.config import update_config
from dxl.learn.core.graph import Graph
from dxl.learn.core.graph_info import GraphInfo
from dxl.learn.core.tensor import Tensor, Constant
from dxl.learn.test import TestCase
# from dxl.learn.core.subgraph_maker import SubgraphPartialMaker, SubgraphMaker, SubgraphMakerTable


class TestGraph(TestCase):
    def assertInfoCorrectlyInitialized(self, g, name):
        self.assertIsInstance(g.info, GraphInfo)
        self.assertNameEqual(g.name, name)
        self.assertNameEqual(g, name)

    def test_info_from_str(self):
        g = Graph('g')
        self.assertInfoCorrectlyInitialized(g, 'g')

    def test_input_info(self):
        g = Graph(GraphInfo('g', 'g_scope'))
        self.assertNameEqual(g, 'g')
        self.assertNameEqual(g.info, 'g')
        self.assertNameEqual(g.info.scope, 'g_scope')

    def test_make_info(self):
        pass

    def test_name_of_graphs(self):
        class TestGraph(Graph):
            def kernel(self):
                self.graphs['subg'] = Graph(self.name / 'subg')

        g = TestGraph('g')
        self.assertNameEqual(g.graphs('subg'), 'g/subg')

    def test_config_of_subgraph(self):
        update_config('g/subg', {'key': 'value'})

        class TestGraph(Graph):
            def kernel(self):
                subg = self.graphs('subg')

        g = Graph(
            'g',
            graphs={
                'subg': lambda g, n: Graph.child_maker(g, n, Graph)
            })
        assert g.subgraph('subg').config('key') == 'value'
        self.assertNameEqual(g.subgraph('subg'), 'g/subg')

    def test_subgraph_maker_directly_construct(self):
        class TestSubGraph(Graph):
            pass

        class TestGraph(Graph):
            def kernel(self):
                self.graphs('subg', TestSubGraph('test_subg'))

        g = TestGraph('g')
        assert isinstance(g.subgraph('subg'), TestSubGraph)
        self.assertNameEqual(g.subgraph('subg').info, 'test_subg')

    def test_access_tensor(self):
        class TestGraph(Graph):
            def kernel(self):
                self.tensors['x'] = Tensor(tf.constant(1, name='x'))

        g = TestGraph('test_g')
        g.make()
        assert isinstance(g.tensor('x'), Tensor)
        self.assertNameEqual(g.tensor('x'), 'test_g/x')

    def test_access_config(self):
        update_config('g', {'key1': 1})
        g = Graph('g')
        assert g.config('key1') == 1

    def test_required(self):
        g = Graph('g')
        with pytest.raises(TypeError):
            g.tensor('x', Graph.required_tensor)

    def test_find(self):
        pass

    def test_run(self):
        pass

    def get_graph_with_item_config(self, item=None, info=None, config=None):
        if info is None:
            info = 'g'

        class TestConfigGraph(Graph):
            def __init__(self, info, *, config, item=None):
                print(self._parse_input_config(config, {'item': item}))
                super().__init__(
                    info,
                    config=self._parse_input_config(config, {'item': item}))

            @classmethod
            def _default_config(cls):
                return {'item': 0}

        return TestConfigGraph(info, item=item, config=config)

    def test_default_config(self):
        g = self.get_graph_with_item_config()
        assert g.config('item') == 0

    def test_config_with_none(self):
        g = self.get_graph_with_item_config(item=1)
        assert g.config('item') == 1

    def test_config_with_none_from_external(self):
        update_config('g', {'item': 1})
        g = self.get_graph_with_item_config()
        assert g.config('item') == 1

    def test_config_with_config(self):
        g = self.get_graph_with_item_config(config={'item': 1})
        assert g.config('item') == 1

    def test_config_with_conflict(self):
        update_config('g', {'item': 1})
        g = self.get_graph_with_item_config(item=1)
        assert g.config('item') == 1

    # def test_subgraph_maker_via_graphs_cls(self):
    #     class TestSubGraph(Graph):
    #         pass

    #     x = Constant(1.0, 'x')

    #     class TestGraph(Graph):
    #         def kernel(self):
    #             self.graphs('sub',
    #                           SubgraphPartialMaker(
    #                               self.info.name / 'sub', tensors={'x': x}))

    #     g = TestGraph('g', graphs={'sub': TestSubGraph})
    #     assert isinstance(g.subgraph('sub'), TestSubGraph)
    #     assert g.subgraph('sub').tensor('x') is x

    # def test_subgraph_maker_via_graphs_subgraph_maker(self):
    #     class TestSubGraph(Graph):
    #         pass

    #     x = Constant(1.0, 'x')

    #     class TestGraph(Graph):
    #         def kernel(self):
    #             self.graphs('sub')

    #     g = TestGraph(
    #         'g',
    #         graphs={
    #             'sub':
    #             SubgraphMaker(TestSubGraph,
    #                           SubgraphPartialMaker('sub', tensors={'x': x}))
    #         })
    #     assert isinstance(g.subgraph('sub'), TestSubGraph)
    #     assert g.subgraph('sub').tensor('x') is x

    # def test_subgraph_maker_via_cls_graphs_table(self):
    #     class TestSubGraph(Graph):
    #         pass

    #     x = Constant(1.0, 'x')

    #     SubgraphMakerTable.register('g/sub', TestSubGraph)

    #     class TestGraph(Graph):
    #         def kernel(self):
    #             self.graphs('sub',
    #                           SubgraphPartialMaker(
    #                               self.info.name / 'sub', tensors={'x': x}))

    #     g = TestGraph('g', )
    #     assert isinstance(g.subgraph('sub'), TestSubGraph)
    #     assert g.subgraph('sub').tensor('x') is x

    # def test_subgraph_maker_via_cls_graphs_table(self):
    #     class TestSubGraph(Graph):
    #         pass

    #     x = Constant(1.0, 'x')

    #     SubgraphMakerTable.register('g/sub', TestSubGraph)

    #     class TestGraph(Graph):
    #         def kernel(self):
    #             self.graphs('sub',
    #                           SubgraphPartialMaker(
    #                               self.info.name / 'sub', tensors={'x': x}))

    #     g = TestGraph('g', )
    #     assert isinstance(g.subgraph('sub'), TestSubGraph)
    #     assert g.subgraph('sub').tensor('x') is x

    # def test_subgraph_maker_quick(self):
    #     class TestSubGraph(Graph):
    #         pass

    #     x = Constant(1.0, 'x')

    #     class TestGraph(Graph):
    #         def kernel(self):
    #             self.graphs('sub',
    #                           self.graphs_partial_maker(
    #                               'sub', tensors={'x': x}))

    #     g = TestGraph('g', graphs={'sub': TestSubGraph})
    #     assert isinstance(g.subgraph('sub'), TestSubGraph)
    #     assert g.subgraph('sub').tensor('x') is x
