class SubgraphMakerFactory:
    _graph_maker_builders = {}

    @classmethod
    def register(cls, path, func):
        cls._graph_maker_builders[str(path)] = func

    @classmethod
    def get(cls, path):
        return cls._graph_maker_builders[str(path)]

    @classmethod
    def reset(cls):
        cls._graph_maker_builders = {}


class SubgraphMakerFinder:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, father_graph, subgraph_name):
        result = SubgraphMakerFactory.get(
            father_graph.info.name / subgraph_name)
        if len(self._args) == 0 and len(self._kwargs) == 0:
            return result(father_graph, subgraph_name)
        else:
            return result(*args, **kwargs)(father_graph, subgraph_name)
