from dxl.learn.core.graph import Graph
from dxl.learn.core.tensor import Tensor
import unittest


class TestGraph(unittest.TestCase):
    def test_config(self):
        g = Graph(config={'some_key': 1})
        assert g.config('some_key') == 1

    def test_config_of_subgraph(self):
        class TestGraph(Graph):
            def kernel(self):
                subg = self.subgraph('subg')

        g = Graph(
            name='g',
            subgraph={'subg': lambda g: Graph(name=g.name / 'subg')},
            config={'subg': {
                'some_key': 1
            }})
        assert subg.config('some_key') == 1

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
        assert str(g.tensor('x').name) == 'test_g/x'
