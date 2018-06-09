from 
class Trainer(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            OBJECTIVE = 'objective'
            TRAIN_STEP = 'train_step'

    def __init__(self, info, objective, optimizer, *, config):
        self.optimizer = optimizer
        super().__init__(
            info, tensors={
                self.KEYS.TENSOR.OBJECTIVE: objective,
            })

    def kernel(self):
        KT = self.KEYS.TENSOR
        self.tensors[KT.TRAIN_STEP] = self.optimizer.minimize(
            self.tensor(KT.OBJECTIVE), )
        self.tensors[KT.MAIN] = self.train_step

    @property
    def objective(self):
        return self.tensor(self.KEYS.TENSOR.OBJECTIVE)

    @property
    def train_step(self):
        return self.tensor(self.KEYS.TENSOR.TRAIN_STEP)

    def learning_rate(self):
        pass

    def decay_learning_rate(self, step=1):
        pass