import unittest
import pytest
import tensorflow as tf
import numpy as np
from dxl.learn.test import TestCase
from dxl.learn.network import Network
from dxl.learn.model.dense import Dense
from dxl.learn.dataset import DatasetFromColumns, PyTablesColumns
from dxl.learn.dataset import Train80Partitioner, DataColumnsPartition
from dxl.learn.test.resource import test_resource_path
from dxl.learn.network.metric import mean_square_error
from dxl.learn.network.trainer.optimizers import RMSPropOptimizer
from dxl.learn.network.trainer import Trainer


class TestNetwork(TestCase):
    CONFIG = {
        'h0_units': 32,
        'h1_units': 10
    }
    DATA_PATH = test_resource_path() / 'dataset' / 'mnist.h5'

    def get_columns(self):
        return DataColumnsPartition(
                PyTablesColumns(self.DATA_PATH, '/train'),
                Train80Partitioner(True))

    def get_dataset(self):
        return DatasetFromColumns(
                'datset',
                self.get_columns(),
                nb_epochs=5,
                batch_size=32,
                is_shuffle=True)
    
    def get_trainer(self):
        return Trainer('trainer', 
                RMSPropOptimizer('optimizer', learning_rate=1e-3))
    
    def get_metrices(self):
        return mean_square_error

    def create_network(self):
        class DNNWith2Layers(Network):
            def kernel(self, inputs):
                x = inputs['x']
                h = self.get_or_create_graph('layer0', 
                        Dense('dense', n_units=self.config('h0_units'),
                              activation='relu'))(x)
                y_ = self.get_or_create_graph('layer1', 
                        Dense('dense', n_units=self.config('h1_units'),
                              activation='relu'))(h)
                return {
                    Network.KEYS.TENSOR.INFERENCE: y_,
                }

        dataset = self.get_dataset()
        network = DNNWith2Layers('mnist', tensors={'x': dataset.tensors['x']},
                    trainer=self.get_trainer(),
                    metrics=self.get_metrices())
        return network
 
    def is_mono_decay(self, data):
        data_pre = data[:-1]
        data_now = data[1:]
        return np.all(data_pre > data_now)

    def test_train(self):
        network = self.create_network()
        with self.test_session() as sess:
            losses = []
            for i in range(1000):
                if i % 100 == 0:
                    l = network.run(network.tensors['objective'])
                    losses.append(l)
                network.train()

        assert self.is_mono_decay(losses)
