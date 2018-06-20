# _*_ encoding: utf-8 _*_

import tensorflow as tf
import numpy as np
from dxl.learn.test import TestCase
from dxl.learn.model.super_resolution import SuperResolution2x, SuperResolutionBlock
from dxl.learn.test import UnitBlock
import pytest


class SRKeys:
    REPRESENTS = 'reps'
    RESIDUAL = 'resi'
    ALIGNED_LABEL = 'aligned_label'
    INTERP = 'interp'
    POI_LOSS = 'poi_loss'
    MSE_LOSS = 'mse_loss'


class TestSuperResolution2x(TestCase):
    def get_input(self):
        return tf.constant(np.ones([2, 100, 100, 3], dtype="float32"))
    
    def make_model(self):
        return UnitBlock("unitblock_test")

    def test_SuperResolution2xDef(self):
        x = self.get_input()
        superRe2x_ins = SuperResolution2x(
            'superRe2x_test',
            {'input': x},
            nb_layers=2,
            filters=5,
            boundary_crop=[4, 4])
        res = superRe2x_ins()
        with self.variables_initialized_test_session() as sess:
            for k, v in res.items():
                y = sess.run(v)

    def test_SuperResolution2x(self):
        x = self.get_input()
        superRe2x_ins = SuperResolution2x(
            'sR',
            inputs={'input': x},
            nb_layers=2,
            filters=5,
            boundary_crop=[4, 4],
            graph=self.make_model())
        res = superRe2x_ins()
        with self.variables_initialized_test_session() as sess:
            for k, v in res.items():
                y = sess.run(v)


class TestSuperResolutionBlock(TestCase):
    def get_input(self):
        x = tf.constant(np.ones([2, 32, 32, 3], dtype="float32"))
        y = tf.constant(np.ones([2, 64, 64, 3], dtype="float32"))
        return (x, y)
    
    def make_model(self):
        return UnitBlock("unitblock_test")

    def test_SuperResolutionBlockDef(self):
        # test default graph
        x, y = self.get_input()
        superReBlk_ins = SuperResolutionBlock(
            'superReBlk_test',
            inputs={
            'input': x,
            'label': y})
        res = superReBlk_ins()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for k, v in res.items():
                y = sess.run(v)

    def test_SuperResolutionBlock(self):
        x, y = self.get_input()
        superReBlk_ins = SuperResolutionBlock(
            'superReBlk_test',
            inputs={
             'input': x,
             'label': y},
            graph=self.make_model())
        res = superReBlk_ins()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for k, v in res.items():
                y = sess.run(v)

if __name__ == '__main__':
    tf.test.main()
