from .config import ConfigurableWithName
from typing import Dict, Callable
from .tensor import Tensor
from .distribute import Host
from .graph_info import GraphInfo
from dxl.fs import Path

import warnings


class Graph(ConfigurableWithName):
    """
      Graph is a collection of Tensors and their opeartions.

      `self.tensors` is a dict of Tensors, which is provided as an interface to Graph.
      Which means one may use ::

          g = Graph(...)
          g.run(key)

      to run corresponding tensors.

      Another useage is to substitute part of graph with other tensors. One may use ::

          g = Graph(tensor_dict, ...)
          g.run(fetch_keys, inputs={k: tensor})

      to substitute (by tf.run(feed_dict={g.tensors[k]: inputs[k].data})) mechanism.


      KEYS:

      - GRAPH:

          - MAIN: Main purpose/effect of graph, thus the one which is fetched by
          by default, and most part of the graph should be evaluated.

      Methods:

      Support using Graph as an "function" on Tensors.





      - tensor(key): -> Tensor

    
    Provide The following methods:
    
    - `g.tensor(key)`
    - `g.subgraph(key)`
    - `g.config(key)`

    Some graphs may support:

    - `g.tensors.key`
    - `g.subgraph.key`
    - `g.config.key`

    """

    class KEYS:
        class DOMAIN:
            TENSOR = 'tensor'
            SUBGRAPH = 'subgraph'

        class TENSOR:
            MAIN = 'main'

        class SUBGRAPH:
            pass

    def __init__(self,
                 name: Path,
                 tensors: Dict[str, Tensor] = None,
                 subgraphs: Dict[str, 'Graph'] = None,
                 graph_info: GraphInfo = None,
                 config: Dict[str, 'Config'] = None):

        super().__init__(name, config=config)
        if subgraphs is None:
            subgraphs = dict()
        self.subgraphs = subgraphs
        if tensors is None:
            tensors = dict()
        self.tensors = tensors
        if graph_info is None:
            graph_info = GraphInfo(name)
        self.graph_info = graph_info
        if self.graph_info.scope is None:
            self.graph_info.scope = self.name
        if self.graph_info._name is None:
            self.graph_info._name = name

    def __hash__(self):
        return hash(self.name)

    def keys(self, domain=None):
        if domain == self.KEYS.DOMAIN.TENSOR:
            return self.tensor_keys()
        if domain == self.KEYS.DOMAIN.SUBGRAPH:
            return self.subgraph_keys()
        if domain is None:
            return tuple(list(self.tensor_keys()) + list(self.subgraph_keys()))
        raise ValueError("Unknown domain {}.".format(domain))

    def tensor_keys(self):
        return self.tensors.keys()

    def subgraph_keys(self):
        return self.subgraphs.keys()

    def values(self):
        return self.tensors.values()

    def items(self):
        return self.tensors.values()

    def __iter__(self):
        return self.tensors.__iter__()

    def tensor(self, key, is_required=False):
        if is_required and not key in self.tensors:
            raise ValueError("Key {} is required but not found.".format(key))
        return self.tensors.get(key)

    def subgraph(self, key):
        return self.subgraphs.get(key)

    def get_subgraph(self,
                     key,
                     subgraph_maker: Callable[['Graph'], 'Graph']
                     or 'Graph' = None):
        """
            Get or create subgraph. Useful when defining graph which is intend to be reused.

            Since one may use ::

                g = Graph('name1')
                reused_subgraphs = select_part_of(g.subgraphs)
                g2 = Graph('name2', subgraphs=reused_subgraphs)

            """
        subgraph = self.subgraph(key)
        if subgraph is None:
            self.subgraphs[key] = subgraph_maker(self)
        return self.subgraphs[key]

    def get_tensor(self, key,
                   tensor_maker: Callable[['Graph'], Tensor] = None):
        """
            """
        tensor = self.tensor(key)
        if tensor is None:
            self.tensors[key] = tensor_maker(self)
        return self.tensors[key]

    def run(self, fetches=None, inputs=None, configs=None):
        """
        run graph with given fetches and inputs.
        if fetches is None, use self.KEYS.TENSOR.MAIN.
        if inputs is a dict, valid inputs will be filtered.
        """
        if fetches is None:
            fetches = self.tensor(self.KEYS.TENSOR.MAIN)
        if inputs is not None:
            valid_inputs = {
                k: inputs[k]
                for k in inputs if k in self.tensor_keys()
            }
        else:
            valid_inputs = dict()
        from .session import ThisSession
        feed_dict = {}
        for k in self.tensor_keys(k):
            if k in valid_inputs:
                feed_dict.update(self.tensor(k), inputs[k])
        return ThisSession.run(feed_dict=feed_dict)

    @classmethod
    def tensorflow_tensor(cls, t):
        warnings.warn(
            DeprecationWarning(
                "Graph.tensorflow_tensor will be deprecated, use dxl.learn.core.tf_tensor instead."
            ))
        import tensorflow as tf
        if isinstance(t, tf.Tensor):
            return t
        if isinstance(t, Tensor):
            return t.data
        else:
            raise TypeError("Can not convert {} to tensorflow_tensor.".format(
                type(t)))
