""" Base class definition """
import abc

from dxl.learn.config import config_with_name


class NodeKeys:
    """
    Pre-defined frequently used keys
    """
    EVALUATE = 'evaluate'
    INFERENCE = 'inference'
    TRAINER = 'trainer'
    SAVER = 'saver'
    LOSS = 'loss'
    MAIN = 'main'
    INPUT = 'input'
    LABEL = 'label'
    OUTPUT = 'output'
    MAIN_MODEL = 'main_model'
    CHILD_MODEL = 'child_model'
    TASK = 'task'


__all__ = ['Graph']


class Graph(abc.ABC):
    """
    Base class of components.

    A `Graph` is an dict like collections of nodes and edges. Nodes of `Graph` can be a tensor another sub `Graph`.
    Edges representing relations of sub graphs/tensors, flow of information.

    A `Graph` is an generalization of `tf.Graph`, which is designed for following features:
        1. An unified interface of `tf.Graph` and general compute graph or operations/procedures;
        2. Seperate config and implementation, use TreeDict for configs, and supports multiple ways of config;
        3. An easy-to-use way of seperate/reuse subgraphs;
        4. Supports an warp of sessions.run/normal python function.
            Please add member method for tasks, and register them to tasks

    Config module:
    Configs module is designed to support externel hierachy configs, thus with name of graph,
    graph can load/search configs. This is designed to reduce complecity of config complex networks.
    To achieve this, use more configurable than directly pass arguments to init functions.
    For sub-models, in most cases do not pass arguments by its parent graph, directly use config system.
    There are some exceptions to this design, for some simple/frequently used blocks, some simple arguments
    may be passed to simplify config procedure. Another simulation is the case with lots of child models and
    they all share simple but same configurations, you may pass the shared arguments by its parent Graph.

    In most cases, Graph should communicate(connect) with others via dict of Tensors/Graph.


    Methods:

    -   as_tensor(self):
        return self.nodes['main'], which is designed for sub_graphs.

    -   get_feed_dict(self, task=None):
        A method which returns a feed_dict, which can be used to update parent (in most cases, the graph which called 
        subgraph.get_feed_dict()) graph's get_feed_dict() or run task.
        Which is used to garantee output nodes (if is Tensor) to be valid under certain tasks, if task is None,
        a feed_dict should be provided so that all nodes are valid.

    -   run(self, task_name, feeds):
        Call a registered function. # TODO: make it different from directly call function, provide default feeds / unified feeds, etc.

    Properties:
        name: Path, name is used for:
            1. loading configs from global config object;
            2. its basename sometimes used for namescope/variable name in TensorFlow;
    """

    class KEYS:
        class TENSOR:
            pass

        class GRAPH:
            pass

        class CONFIG:
            pass

    def __init__(self, name):
        self.config = config_with_name(name)
        self.name = name
        self.tasks = {}
        self.graphs = {}
        self.tensors = {}
        self.is_built = False

    def __hash__(self):
        return self.name.__hash__()

    def keys(self):
        return self.tasks.keys()

    def values(self):
        return self.tasks.values()

    def items(self):
        return self.tasks.items()

    def __iter__(self):
        return self.tasks.__iter__()

    def make(self):
        if self.is_built:
            return
        self.kernel()
        self.is_built = True

    @abc.abstractmethod
    def kernel(self):
        pass
