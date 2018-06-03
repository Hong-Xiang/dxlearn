class SubgraphMakerFactory:
    _graph_maker_builders = {}

    @classmethod
    def register(cls, path, func):
        self._graph_maker_builders[str(path)] = func

    @classmethod
    def get(cls, path):
        return self._graph_maker_builders[str(path)]

    @classmethod
    def reset(cls):
        cls._graph_maker_builders = {}