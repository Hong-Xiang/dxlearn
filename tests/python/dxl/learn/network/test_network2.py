# from dxl.learn import Network, Dense, ReLU, MeanSquareError
# from dxl.learn.dataset import MNIST
import unittest
import tensorflow as tf
import numpy as np
import pytest


@pytest.mark.skip(reason='not impl yet')
def create_2_layers_dnn_for_mnist():
    class DNNWith2Layers(Network):
        def kernel(self, inputs):
            x = inputs['x']
            h = self.graphs('layer0', lambda g: Dense(x, self.config('hidden_units')[0]), activation=ReLU)(x)
            h = self.graphs('layer1', lambda g: Dense(h, self.config('hidden_units')[1]), activation=ReLU)(h)
            label = inputs['y']
            y_ = self.graphs('layer1', lambda g: Dense(h, self.config('hidden_units')[2]))(h)
            l = MeanSquareError(y_, inputs['y'])
            return {
                Network.KEYS.TENSOR.INFERENCE: y_,
                Network.KEYS.TENSOR.LOSS: l
            }

    dataset = MNIST(partition='train')
    network = DNNWith2Layers(inputs={'x': dataset.tensor('x')})
    return network


def is_mono_decay(data):
    data_pre = data[:-1]
    data_now = data[1:]
    return np.all(data_pre > data_now)


def create_mnist_train_dataset():
    pass


@pytest.mark.skip(reason='not impl yet')
class TestNetwork(tf.test.TestCase):
    def test_train(self):
        network = create_2_layers_dnn_for_mnist()
        with self.test_session() as sess:
            losses = []
            for i in range(1000):
                if i % 100 == 0:
                    l = network.run(network.KEYS.LOSS)
                    losses.append(l)
                network.train()
        assert is_mono_decay(losses)
