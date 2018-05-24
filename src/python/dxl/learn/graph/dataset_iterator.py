class DatasetIterator(Graph):
    def __init__(self, dataset, config):
        self.dataset = dataset
    
    def save(self):
        path = self.config('save')['path']
        for _ in range(self.dataset.capacity()):
            s = ThisSession.run(self.dataset['x_l']) 
            # save s to filesystem

