from dxl.learn.core import Graph
from dxl.learn.tensor.global_step import GlobalStep


class Trainer(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            OBJECTIVE = 'objective'
            TRAIN_STEP = 'train_step'

        class GRAPH(Graph.KEYS.GRAPH):
            OPTIMIZER = 'optimier'

    def __init__(self, info, optimizer, objective=None):
        super().__init__(
            info,
            tensors={
                self.KEYS.TENSOR.OBJECTIVE: objective,
            },
            graphs={self.KEYS.GRAPH.OPTIMIZER: optimizer})

    def kernel(self, inputs=None):
        KT, KS = self.KEYS.TENSOR, self.KEYS.GRAPH
        objective = self.get_or_create_tensor(KT.OBJECTIVE,
                                              inputs[KT.OBJECTIVE])
        self.graphs[KS.OPTIMIZER].make()
        self.tensors[KT.TRAIN_STEP] = self.graphs[KS.OPTIMIZER].minimize(
            objective, global_step=GlobalStep().data)
        self.tensors[KT.MAIN] = self.train_step

    @property
    def objective(self):
        return self.tensors[self.KEYS.TENSOR.OBJECTIVE]

    @property
    def train_step(self):
        return self.tensors[self.KEYS.TENSOR.TRAIN_STEP]

    def learning_rate(self):
        return self.graphs[self.KEYS.GRAPH.OPTIMIZER].learning_rate

    @property
    def decay_learning_rate(self, step=1):
        return self.graphs[self.KEYS.GRAPH.OPTIMIZER].decay_learning_rate


class TrainerV2(Graph):
    class KEYS(Graph.KEYS):
        class GRAPH(Graph.KEYS.GRAPH):
            OPTIMIZER = 'optimizer'

    def __init__(self, info, optimizer):
        super().__init__(info)
        self.optimizer = optimizer
