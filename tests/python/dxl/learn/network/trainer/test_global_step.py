import pytest
import tensorflow as tf
import numpy as np
from dxl.learn.test import TestCase
from dxl.learn.network.trainer.global_step import GlobalStep

class TestGlobalStep(TestCase):
    def make_global_step(self):
        return GlobalStep()

    def imitate_train(self, g_step):
        g_step.increased()

    def test_global_step(self):
        train_step = 2
        g_step = self.make_global_step()
        with self.variables_initialized_test_session() as sess:
            for i in range(train_step):
                self.imitate_train(g_step)
                gt = sess.run(g_step)
                self.assertEqual(gt, i)
