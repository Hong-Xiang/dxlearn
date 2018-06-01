from ..graph import Graph


class GraphFromDict(Graph):
    def __init__(self, nodes, *, name='dict_graph', **config):
        super().__init__(name, **config)
        for n, v in nodes.items():
            self.register_node(n, v)


class GraphProxy(Graph):
    def __init__(self, origin, proxy_map, *, name='proxy', **config):
        """
        Args:
            proxy_map:  dict of new_name: old_name
        """
        super().__init__(name, **config)
        for n, v in proxy_map:
            self.register_node(n, origin[v])
