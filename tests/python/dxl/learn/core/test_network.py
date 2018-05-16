from dxl.learn import Network, Dense, ReLU, MSE
from dxl.learn.dataset import MNIST
import unittest

def create_2_layers_dnn_for_mnist():
    class DNN2Layer(Network):
        def kernel(self, inputs):
            x = inputs['x']
            h = self.subgraph('layer0', lambda g: Dense(x, self.config('hidden_units')[0]), activation=ReLU)(x)
            h = self.subgraph('layer1', lambda g: Dense(x, self.config('hidden_units')[1]), activation=ReLU)(h)
            label = inputs['y']
            y_ = self.subgraph('layer1', lambda g: Dense(x, self.config('hidden_units')[2]))(h)
            l = 
            return {'pred': }
    dataset = MNIST(partition='train')
    network = DNN2Layer(inputs={'x': dataset.tensor('x')})

   

class TestNetwork(unittest.TestCase):
    def test_2_layers_dnn(self):
