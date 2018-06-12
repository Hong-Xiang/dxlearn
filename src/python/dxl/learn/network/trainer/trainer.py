from dxl.learn.core import Graph


class Trainer(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            OBJECTIVE = 'objective'
            TRAIN_STEP = 'train_step'

        class SUBGRAPH(Graph.KEYS.SUBGRAPH):
            OPTIMIZER = 'optimier'

    def __init__(self, info, objective, optimizer, *, config):
        super().__init__(
            info,
            tensors={
                self.KEYS.TENSOR.OBJECTIVE: objective,
            },
            graphs={self.KEYS.SUBGRAPH.OPTIMIZER: optimizer})

    def kernel(self):
        KT, KS = self.KEYS.TENSOR, self.KEYS.SUBGRAPH
        self.tensors[KT.TRAIN_STEP] = self.graphs(KS.OPTIMIZER).minimize(
            self.tensor(KT.OBJECTIVE))
        self.tensors[KT.MAIN] = self.train_step

    @property
    def objective(self):
        return self.tensor(self.KEYS.TENSOR.OBJECTIVE)

    @property
    def train_step(self):
        return self.tensor(self.KEYS.TENSOR.TRAIN_STEP)

    def learning_rate(self):
        return self.graphs(self.KEYS.SUBGRAPH.OPTIMIZER).learning_rate

    @property
    def decay_learning_rate(self, step=1):
        return self.graphs(self.KEYS.SUBGRAPH.OPTIMIZER).decay_learning_rate
