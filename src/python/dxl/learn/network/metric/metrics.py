import tensorflow as tf
from dxl.learn.core import Graph

class Metrics(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            LABEL = 'label'
            DATA = 'data'
        
        class Graph(Graph.KEYS.GRAPH):
            METRIC = 'metric'

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR

