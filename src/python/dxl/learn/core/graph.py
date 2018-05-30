from .config import ConfigurableWithName
from typing import Dict, Callable, TypeVar
from .tensor import Tensor
from .distribute import Host
from .graph_info import GraphInfo
from pathlib import Path

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

        class TENSOR:
            MAIN = 'main'

        class SUBGRAPH:
            pass

        class CONFIG:
            pass

    def __init__(self,
                 info: TypeVar('ConvertableToInfo', Path, GraphInfo),
                 config: Dict[str, 'Config'] = None,
                 tensors: Dict[str, Tensor] = None,
                 subgraphs: Dict[str, 'Graph'] = None):
        super().__init__(self._name_for_configurable(info), config=config)
        self.info = self.make_info(info)
        self.subgraphs = subgraphs or dict()
        self.tensors = tensors or dict()
        self._make_kernel_with_scope()

    def _name_for_configurable(self, info):
        if isinstance(info, (str, Path)):
            return info
        if isinstance(info, GraphInfo):
            return info.name
        raise TypeError("Invalid name or graph info: {}.".format(info))

    def make_info(self, info):
        if isinstance(info, (Path, str)):
            return self.default_info(info)
        if not isinstance(info, GraphInfo):
            raise TypeError("Invalid info type for {}.".format(info))
        return info

    def _make_kernel_with_scope(self):
        with self.info.variable_scope():
            self.kernel()

    def kernel(self):
        """
        Users may overwrite this function to construct graph.
        """
        pass

    def default_info(self, name):
        """
        User may overwrite this function to provide default GraphInfo
        """
        return GraphInfo(name)

    def __hash__(self):
        return hash(str(self.info.name))

    def keys(self, domain=None):
        warnings.warn(DeprecationWarning())
        if domain == self.KEYS.DOMAIN.TENSOR:
            return self.tensor_keys()
        if domain == self.KEYS.DOMAIN.SUBGRAPH:
            return self.subgraph_keys()
        if domain is None:
            return tuple(list(self.tensor_keys()) + list(self.subgraph_keys()))
        raise ValueError("Unknown domain {}.".format(domain))

    @classmethod
    def child_maker(self, g, name, constructor):
        return constructor(g.info.child(name))

    def tensor_keys(self):
        warnings.warn(DeprecationWarning('Use self.tensors.keys() instead.'))
        return self.tensors.keys()

    def subgraph_keys(self):
        warnings.warn(DeprecationWarning('Use self.subgraphs.keys() instead.'))
        return self.subgraphs.keys()

    def values(self):
        warnings.warn(DeprecationWarning())
        return self.tensors.values()

    def items(self):
        warnings.warn(DeprecationWarning())
        return self.tensors.values()

    def __iter__(self):
        warnings.warn(DeprecationWarning())
        return self.tensors.__iter__()

    @classmethod
    def raise_error(g, key, expected_type):
        raise TypeError('Required key {} of {}.{} is not found.'.format(
            key, g, expected_type))

    @classmethod
    def required_tensor(cls):
        return lambda g, n: raise_error(g, n, 'tensor')

    @classmethod
    def required_subgraph(cls):
        return lambda g, n: raise_error(g, n, 'subgraph')

    def _get_or_create_item(self, collection, key, expected_type, maker):
        if not collection.get(key) is None and isinstance(
                collection.get(key), expected_type):
            return collection.get(key)
        if maker is None and collection.get(key) is not None:
            maker = collection.get(key)
        if maker is not None:
            item = maker(self, key)
            collection[key] = item
        return collection.get(key)

    def tensor(self, key, maker=None):
        return self._get_or_create_item(self.tensors, key, Tensor, maker)

    def subgraph(self, key, maker=None):
        return self._get_or_create_item(self.subgraphs, key, Graph, maker)

    def get_tensor(self, key,
                   tensor_maker: Callable[['Graph'], Tensor] = None):
        """
            """
        warnings.warn(DeprecationWarning('Use self.tensor.'))
        tensor = self.tensor(key)
        if tensor is None:
            self.tensors[key] = tensor_maker(self)
        return self.tensors[key]

    def parse_names_maybe(self, data):
        warnings.warn(DeprecationWarning())
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
                for k in inputs if k in self.tensors.keys()
            }
        else:
            valid_inputs = dict()
        from .session import default_session
        feed_dict = {}
        for k in self.tensor_keys(k):
            if k in valid_inputs:
                feed_dict.update(self.tensor(k), inputs[k])
        return default_session().run(feed_dict=feed_dict)

    # @property
    # def info(self):
    #     return self.graph_info

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
