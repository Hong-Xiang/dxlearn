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


class SubgraphPartialMaker:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, cls):
        return cls(*self._args, **self._kwargs)


class SubgraphMaker:
    def __init__(self, cls, partial_maker):
        self._cls = cls
        self._partial_maker = partial_maker

    def __call__(self):
        return self._partial_maker(self._cls)


class SubgraphMakerTable:
    _data = {}

    @classmethod
    def register(cls, path, maker):
        cls._data[str(path)] = maker

    @classmethod
    def get(cls, path):
        return cls._data.get(str(path))

    @classmethod
    def reset(cls):
        cls._data = {}