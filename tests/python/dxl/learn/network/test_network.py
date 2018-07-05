import unittest
import pytest
import tensorflow as tf
import numpy as np
from dxl.learn.test import TestCase
from dxl.learn.network import Network
from dxl.learn.model.dense import Dense
from dxl.learn.core import Model
from dxl.learn.dataset import DatasetFromColumns, PyTablesColumns
from dxl.learn.dataset import Train80Partitioner, DataColumnsPartition
from dxl.learn.test.resource import test_resource_path
from dxl.learn.network.metric import mean_square_error
from dxl.learn.network.trainer.optimizers import RMSPropOptimizer
from dxl.learn.network.trainer import Trainer
from dxl.learn.core import ThisSession, Tensor


class TestNetwork(TestCase):
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

    def get_optimizer(self):
        return RMSPropOptimizer('optimizer', learning_rate=1e-3)

    def get_trainer(self):
        return Trainer('trainer',
                       self.get_optimizer())

    def get_metrices(self):
        return mean_square_error

    def create_network(self):
        class DNNWith2Layers(Model):
            def kernel(self, inputs):
                x = inputs['image']
                x = Tensor(tf.cast(x.data, tf.float32))
                label = inputs['label']
                label = Tensor(tf.cast(label.data, tf.float32))
                label = tf.reshape(label.data, (32, 1))

                h = self.get_or_create_graph('layer0',
                                             Dense(
                                                 'dense0',
                                                 n_units=32,
                                                 activation='relu'))(
                                                     tf.layers.flatten(x.data))
                y_ = self.get_or_create_graph('layer1',
                                              Dense(
                                                  'dense1',
                                                  n_units=10,
                                                  activation='relu'))(h)

                y = self.get_or_create_graph('layer3',
                                             Dense(
                                                 'dense2',
                                                 n_units=10,
                                                 activation='relu'))(label)
                return {
                    'inference' : y_,
                    'label' : y
                }

        dataset = self.get_dataset()
        dataset.make()
        model = DNNWith2Layers('mnist', tensors=dataset.tensors)

        model = Network('minst', model)
        model.bind(metrics=self.get_metrices())
        model.bind(optimizer=self.get_optimizer())
        model.bind('trainer0', model.tensors['label'], model.tensors['inference'])

        return model

    def is_mono_decay(self, data):
        data_pre = data[:-1]
        data_now = data[1:]
        return np.all(data_pre > data_now)

    def test_train(self):
        network = self.create_network()
        network.make()
        with self.variables_initialized_test_session() as sess:
            ThisSession.set_session(sess)
            losses = []
            for i in range(100):
                if i % 10 == 0:
                    l = sess.run(network.tensors['trainer0_objective'])
                    losses.append(l)
                network.train()

        assert self.is_mono_decay(losses)
