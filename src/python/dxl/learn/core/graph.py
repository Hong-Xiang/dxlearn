from .config import ConfigurableWithName
from typing import Dict, Callable, TypeVar
from .tensor import Tensor
from .info import GraphInfo
from pathlib import Path

import warnings

# from .subgraph_maker import SubgraphPartialMaker, SubgraphMaker, SubgraphMakerTable


class Graph(ConfigurableWithName):
    """
    Graph is a collections of Tensors, graphs, configs and info.

    Features for config:

    - provide access to default / external config by `info.name`;
    - provide self.config(key) search-like config getter, from config of father graph.
    - provide _default_config() classmethod;


    Features for tenosrs:
    - provide default collections of holding tensors: self.tensors, main interface of graph.

    `self.tensors` is a dict of Tensors, which is provided as an interface to Graph.
    Which means one may use ::

        g = Graph(...)
        g.run(key) 
        sess.run(g.tensor(key))
        
    to run corresponding tensors.

    Another useage is to substitute part of graph with other tensors. One may use ::

        g = SomeGraph(tensor_dict, ...)
        g.tensors = {'x': Tensor}
        g.run(key, feeds={'x': tensor, 'y': tensor2})
        g.x == g.tensor('x')
        g.run(key, feeds={g.x: tensor, g.y: tensor})

    which is equivilant to ::

        sess.run(g.tensor(key).data, feeds={g.tensor(k).data:tensor for k in ['x', 'y']})


    KEYS:

    - `TENSOR`:
        - `MAIN`: Main purpose/effect of graph, thus the one which is fetched by
        by default, thus `g.run()` is eqvilant to `g.run(g.KEYS.TENSOR.MAIN)`.

        - tensor(key): -> Tensor


    Provide The following methods:

    - `g.tensor(key)`
    - `g.graph(key)`
    - `g.config(key)`

    - `g.run(key)`


    # graph maker design

    Since our library targeting easily reuse and substitution of sub-graph,
    there would be four common cases when constructing Graph with sub-graphs.

    1. father graph is not going to be reused (e.g. for Graphs), subgraph is fixed
    2. father graph is going to be reused (e.g. for Model), subgraph is fixed
    3. father graph is not going to be reused, subgraph is configurable
    4. father graph is going to be reused, subgraph is configurable

    For case 1:
    just directly code it in kernel:
    ``` python
    def kernel(self):
        x = self.tensor('input')
        subg = SomeGraph(self.info.child_scope('sub'), tensors={'x': x})
        subg.make()
        y = subg.tensor('y')
    ```

    For case 2:
    Use `graphs` collection.
    ``` python
    # subg is model
    def kernel(self):
        x = self.tensor('input')
        subg = self.graph('sub', Conv2D(filters=32))
        y = subg(x)
    ```

    For case 3:
    ``` python
    def kernel(self):
        x = self.tensor('input')
        subg = self.graph('sub')
        subg.tensors['x'] = x
        subg.make()
        y = subg.tensor('y')
    ```

    For case 4:
    ``` python
    def kernel(self):
        x = self.tensor('input')
        subg = self.graph('sub')
        y = subg(x)
    ```
    """

    class KEYS:
        class DOMAIN:
            TENSOR = 'tensor'
            SUBGRAPH = 'subgraph'

        class TENSOR:
            MAIN = 'main'

        class GRAPH:
            pass

        class CONFIG:
            pass

    def __init__(self,
                 info: TypeVar('ConvertableToInfo', Path, GraphInfo),
                 config: Dict[str, 'Config'] = None,
                 tensors: Dict[str, Tensor] = None,
                 graphs: Dict[str, 'Graph'] = None):
        super().__init__(self._name_for_configurable(info), config=config)
        self.info = self.make_info(info)
        self.graphs = graphs or dict()
        self.tensors = tensors or dict()
        self.is_made = False

    def make(self, inputs=None):
        if not self.is_made:
            self._make_kernel_with_scope(inputs)
            self.is_made = True

    def _name_for_configurable(self, info):
        if isinstance(info, (str, Path)):
            return info
        if isinstance(info, GraphInfo):
            return info.name
        raise TypeError("Invalid name or graph info: {}.".format(info))

    def make_info(self, info):
        if info is None:
            return info
        if isinstance(info, (Path, str)):
            return self._default_info(info)
        if not isinstance(info, GraphInfo):
            raise TypeError("Invalid info type for {}.".format(info))
        return info

    def _make_kernel_with_scope(self, inputs=None):
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, Dict):
            inputs = {self.KEYS.TENSOR.MAIN: inputs}
        for k, v in self.tensors.items():
            if v is not None and inputs.get(k) is None:
                inputs[k] = v

        with self.info.variable_scope():
            self.kernel(inputs)

    def kernel(self, inputs=None):
        """
        Users may overwrite this function to construct graph.
        """
        pass

    def _default_info(self, name):
        """
        User may overwrite this function to provide default GraphInfo
        """
        return GraphInfo(name)

    def __hash__(self):
        return hash(str(self.info.name))

    # def keys(self, domain=None):
    #     warnings.warn(DeprecationWarning())
    #     if domain == self.KEYS.DOMAIN.TENSOR:
    #         return self.tensor_keys()
    #     if domain == self.KEYS.DOMAIN.SUBGRAPH:
    #         return self.graphs_keys()
    #     if domain is None:
    #         return tuple(list(self.tensor_keys()) + list(self.graphs_keys()))
    #     raise ValueError("Unknown domain {}.".format(domain))

    # @classmethod
    # def child_maker(self, g, name, constructor):
    #     return constructor(g.info.child_scope(name))

    # def tensor_keys(self):
    #     warnings.warn(DeprecationWarning('Use self.tensors.keys() instead.'))
    #     return self.tensors.keys()

    # def subgraph_keys(self):
    #     warnings.warn(DeprecationWarning('Use self.graphs.keys() instead.'))
    #     return self.graphs.keys()

    # def values(self):
    #     warnings.warn(DeprecationWarning())
    #     return self.tensors.values()

    # def items(self):
    #     warnings.warn(DeprecationWarning())
    #     return self.tensors.values()

    # def __iter__(self):
    #     warnings.warn(DeprecationWarning())
    #     return self.tensors.__iter__()

    # @classmethod
    # def raise_error(g, key, expected_type):
    #     raise TypeError('Required key {} of {}.{} is not found.'.format(
    #         key, g, expected_type))

    # @classmethod
    # def required_tensor(cls):
    #     return lambda g, n: raise_error(g, n, 'tensor')

    # @classmethod
    # def required_graph(cls):
    #     return lambda g, n: raise_error(g, n, 'graph')

    # def _get_or_create_item(self, collection, key, expected_type, maker):
    #     result = self._make_subgraph_v2(key, maker)
    #     if result is not None:
    #         return result
    #     if not collection.get(key) is None:
    #         if isinstance(collection.get(key), expected_type):
    #             return collection.get(key)
    #         if isinstance(collection.get(key), (list, tuple)) and all(
    #                 map(lambda x: isinstance(x, expected_type),
    #                     collection.get(key))):
    #             return collection.get(key)
    #         if isinstance(collection.get(key), dict) and all(map(lambda k: isinstance(collection.get(key)[k], expected_type), collection.get(key))):
    #             return collection.get(key)
    #     if maker is None and collection.get(key) is not None:
    #         maker = collection.get(key)
    #     if maker is not None:
    #         if isinstance(maker, expected_type):
    #             item = maker
    #         else:
    #             item = maker(self, key)
    #         collection[key] = item
    #     return collection.get(key)

    # def _make_subgraph_v2(self, key, maker):
    #     value = self.graphs.get(key,
    #                                SubgraphMakerTable.get(
    #                                    self.info.name / key))
    #     if isinstance(value, Graph):
    #         return value
    #     if isinstance(value, SubgraphMaker):
    #         self.graphs[key] = value()
    #         return self.graphs(key)
    #     if isinstance(value, type) and issubclass(value, Graph) and isinstance(
    #             maker, SubgraphPartialMaker):
    #         self.graphs[key] = SubgraphMaker(value, maker)()
    #         return self.graphs(key)
    #     if value is None and isinstance(maker, SubgraphMaker):
    #         self.graphs[key] = maker()
    #         return self.graphs(key)
    # raise TypeError(
    #     "Can't find any solution to make subgraph: in self.graphs: {}, in table: {}, maker: {}".
    #     format(
    #         self.graphs.get(key), SubgraphMakerTable.get(key), maker))

    # def tensor(self, key, maker=None):
    # return self._get_or_create_item(self.tensors, key, Tensor, maker)

    # def subgraph_partial_maker(self, key, *args, **kwargs):
    #     return SubgraphPartialMaker(self.info.name / key, *args, **kwargs)
    # def tensor(self, key):
    #     return self.tensors.get(key)

    # def graph(self, key):
    #     return self.graphs.get(key)

    def get_or_create_tensor(self, key, create=None):
        result = self.tensors.get(key)
        if result is None and create is not None:
            self.tensors[key] = create
            result = create

        return result

    def get_or_create_graph(self, key, create=None):
        result = self.graphs.get(key)
        if result is None and create is not None:
            self.graphs[key] = create
            result = create

        return result

    # def get_tensor(self, key,
    #                tensor_maker: Callable[['Graph'], Tensor] = None):
    #     """
    #         """
    #     warnings.warn(DeprecationWarning('Use self.tensor.'))
    #     tensor = self.tensor(key)
    #     if tensor is None:
    #         self.tensors[key] = tensor_maker(self)
    #     return self.tensors[key]

    # def parse_names_maybe(self, data):
    #     warnings.warn(DeprecationWarning())
    #     if isinstance(data, tf.Tensor):
    #         return data
    #     if isinstance(data, Tensor):
    #         return data.data
    #     if isinstance(data, (Path, str)):
    #         name = Path(data)
    #         if len(name.parts) == 1:
    #             return self.tensor(str(name))
    #         else:
    #             pass

    def find(self, name):
        """
        Helper function to get tensor with deep path.
        If name is a normal name, thus no '/' included, returns self.tensor(name);
        If name has '/' inlcuded, like 'a/x', return self.graphs('a').tensor('x')
        """
        if len(Path(name).parts) == 1:
            return self.tensors[str(name)]
        return self.graphs(Path(name).parts[0]).find('/'.join(
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
            fetches = self.tensors[self.KEYS.TENSOR.MAIN]
        if inputs is not None:
            valid_inputs = {
                k: inputs[k]
                for k in inputs if k in self.tensors.keys()
            }
        else:
            valid_inputs = dict()
        from .session import default_session
        feed_dict = {}
        for k in self.tensors.keys():
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


# class MainGraph(Graph):
#     kernel_func = None

#     def __init__(self, config):
#         super().__init__(None, config=config)

#     def kernel(self):
#         self.kernel_func()

#     @classmethod
#     def make(cls):
#         self.kernel()
#         self._is_made = True

#     @classmethod
#     def set_kernel(cls, func):
#         cls.kernel_func = func
