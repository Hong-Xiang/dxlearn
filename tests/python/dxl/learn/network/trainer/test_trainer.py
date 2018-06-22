import pytest
import tensorflow as tf
import numpy as np
from dxl.learn.network.trainer.optimizers import RMSPropOptimizer
from dxl.learn.network.trainer import Trainer
from dxl.learn.test import TestCase

class TestTrainer(TestCase):
    def get_objective(self):
        return tf.Variable(np.ones([2, 50, 50, 3]), tf.float32)

    def get_optimizer(self):
        return RMSPropOptimizer('optimizer', learning_rate=1e-3)

    def make_trainer(self):
        return Trainer('trainer',
                       optimizer=self.get_optimizer(),
                       objective=self.get_objective())

    def test_objective(self):
        trainer = self.make_trainer()
        trainer.make()
        shape = self.get_objective().shape.as_list()
        expect_shape = trainer.objective.shape.as_list()
        self.assertAllEqual(shape, expect_shape)

    def test_train_step(self):
        trainer = self.make_trainer()
        trainer.make()
        train_step = trainer.train_step
        with self.variables_initialized_test_session() as sess:
            sess.run(train_step)

            