from .config import ConfigurableWithName
from typing import Dict, Callable
from .tensor import Tensor
from .distribute import Host
from .graph_info import GraphInfo
from dxl.fs import Path

import warnings


class Graph(ConfigurableWithName):
    """
    Graph is a collections of Tensors, subgraphs and configs.

    Features for config:

    - provide access to default / external config by info.name;
    - provide default_config() classmethod;
    - provide self.config(key) rearch-like config getter.


    Features for tenosrs:
    - provide default collections of holding tensors: self.tensors
    - access to these 

    `self.tensors` is a dict of Tensors, which is provided as an interface to Graph.
    Which means one may use ::

        g = Graph(...)
        g.run(key)

    to run corresponding tensors.

    Another useage is to substitute part of graph with other tensors. One may use ::

        g = Graph(tensor_dict, ...)
        g.run(key, feeds={k: tensor})

    which is equivilant to ::

        tf.run(g.tensor(key).data, feeds={g.tensor[k]:tensor})


    KEYS:

    - `TENSOR`:
        - `MAIN`: Main purpose/effect of graph, thus the one which is fetched by
        by default, thus `g.run()` is eqvilant to `g.run(g.KEYS.TENSOR.MAIN)`.

      Methods:

      Support using Graph as an "function" on Tensors.





      - tensor(key): -> Tensor


    Provide The following methods:

    - `g.tensor(key)`
    - `g.subgraph(key)`
    - `g.config(key)`


    """

    class KEYS:
        class DOMAIN:
            TENSOR = 'tensor'
            SUBGRAPH = 'subgraph'
            CONFIG = 'config'

        class TENSOR:
            MAIN = 'main'

        class SUBGRAPH:
            pass

        class CONFIG:
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
            graph_info = self.default_info()
        self.graph_info = graph_info
        if self.graph_info.scope is None:
            self.graph_info.scope = self.name
        if self.graph_info._name is None:
            self.graph_info._name = name
        self.build()

    def build(self):
        with self.info.variable_scope():
            self.kernel()

    def kernel(self):
        """
        Users may overwrite this function to construct graph.
        """
        pass

    def default_info(self):
        """
        User may overwrite this function to provide default GraphInfo
        """
        return GraphInfo(self.name)

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
        # Old version
        # TODO: Clear it and fix all usage.
        if isinstance(is_required, bool):
            if is_required and not key in self.tensors:
                raise ValueError(
                    "Key {} is required but not found.".format(key))
            return self.tensors.get(key)
        else:
            t = self.tensors.get(key)
            if t is None and isinstance(is_required, callable):
                t = is_required(self)
            return t

    def subgraph(self,
                 key: str = None,
                 subgraph_maker: Callable[['ParentGraph'], 'subGraph'] = None):
        subgraph = self.subgraphs.get(key)
        if not isinstance(subgraph, Graph):
            if isinstance(subgraph, Callable):
                subgraph = subgraph(self)
        elif subgraph is None:
            self.subgraphs[key] = subgraph_maker(self, key)
            subgraph = self.subgraphs.get(key)
        return subgraph

    def get_tensor(self, key,
                   tensor_maker: Callable[['Graph'], Tensor] = None):
        """
            """
        tensor = self.tensor(key)
        if tensor is None:
            self.tensors[key] = tensor_maker(self)
        return self.tensors[key]

    def parse_names_maybe(self, data):
        if isinstance(data, tf.Tensor):
            return data
        if isinstance(data, Tensor):
            return data.data
        if isinstance(data, (Path, str)):
            name = Path(data)
            if len(name.parts) == 1:
                return self.tensor(str(name))
            else:
                pass

    def find(self, name):
        """
        Helper function to get tensor with deep path.
        If name is a normal name, thus no '/' included, returns self.tensor(name);
        If name has '/' inlcuded, like 'a/x', return self.subgraph('a').tensor('x')
        """
        if len(Path(name).parts) == 1:
            return self.tensor(str(name))
        return self.subgraph(Path(name).parts[0]).find('/'.join(
            Path(name).parts[1:]))

    def run(self, fetches=None, feeds=None):
        """
        run graph with given fetches and inputs.
        if fetches is None, use self.KEYS.TENSOR.MAIN.
        if inputs is a dict, valid inputs will be filtered.

        Graph.run provide the following functionalities:
        1. run by name, when g.run('x') is actually calling tf.run(g.tensor('x'))
        2. add feeds by name.
        """
        inputs = feeds
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

    @property
    def info(self):
        return self.graph_info

    @classmethod
    def tensorflow_tensor(cls, t):
        warnings.warn(
            "Graph.tensorflow_tensor will be deprecated, use dxl.learn.core.tf_tensor instead.",
            DeprecationWarning)
        import tensorflow as tf
        if isinstance(t, tf.Tensor):
            return t
        if isinstance(t, Tensor):
            return t.data
        else:
            raise TypeError("Can not convert {} to tensorflow_tensor.".format(
                type(t)))
