from src.python.dxl.learn.core.config import ConfigurableWithName
from typing import Dict, Callable, TypeVar, Union
from src.python.dxl.learn.core.tensor import Tensor
from src.python.dxl.learn.core.info import GraphInfo
from pathlib import Path

import warnings


# from .subgraph_maker import SubgraphPartialMaker, SubgraphMaker, SubgraphMakerTable

# TODO: remove


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
                 info: Union[Path, GraphInfo],
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
