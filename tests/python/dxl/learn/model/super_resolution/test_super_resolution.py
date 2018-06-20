# _*_ encoding: utf-8 _*_

import tensorflow as tf
import numpy as np
from dxl.learn.test import TestCase
from dxl.learn.model.super_resolution import SuperResolution2x, SuperResolutionBlock
from dxl.learn.model.cnn import ResidualIncept, ResidualStackedConv
from dxl.learn.model.cnn import StackedResidualIncept, StackedResidualConv

import pytest

class SRKeys:
    REPRESENTS = 'reps'
    RESIDUAL = 'resi'
    ALIGNED_LABEL = 'aligned_label'
    INTERP = 'interp'
    POI_LOSS = 'poi_loss'
    MSE_LOSS = 'mse_loss'


class SuperResolution2xTest(TestCase):
    def test_SuperResolution2xDef(self):
        # test default sub_block
        x = np.random.randint(0, 255, [2, 32, 32, 3])
        superRe2x_ins = SuperResolution2x(
            'superRe2x_test',
            inputs={'input': tf.constant(x, dtype='float32')},
            nb_layers=2,
            filters=5,
            boundary_crop=[4, 4])
        res = superRe2x_ins()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for key, y in res.items():
                y = sess.run(y)

    def test_SuperResolution2xInp(self):
        # test input sub_block
        x = np.random.randint(0, 255, [2, 32, 32, 3])
        nb_layers = 2
        ratio = 0.3
        rsc_ins = ResidualStackedConv(
            'sR/src/rsc',
            inputs=tf.constant(x, dtype='float32'),
            ratio=ratio)
        src_ins = StackedResidualConv(
            'sR/src',
            inputs=tf.constant(x, dtype='float32'),
            nb_layers=nb_layers,
            sub_block=rsc_ins)
        superRe2x_ins = SuperResolution2x(
            'sR',
            inputs={'input': tf.constant(x, dtype='float32')},
            nb_layers=2,
            filters=5,
            boundary_crop=[4, 4],
            sub_block=src_ins)
        res = superRe2x_ins()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for key, y in res.items():
                y = sess.run(y)


class SuperResolutionBlockTest(TestCase):
    def test_SuperResolutionBlockDef(self):
        # test default sub_block
        x = np.random.randint(0, 255, [2, 32, 32, 3])
        l = np.random.randint(0, 255, [2, 64, 64, 3])
        superReBlk_ins = SuperResolutionBlock(
            'superReBlk_test',
            inputs={
                'input': tf.constant(x, tf.float32),
                'label': tf.constant(l, tf.float32)
            })
        res = superReBlk_ins()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for key, y in res.items():
                y = sess.run(y)


if __name__ == '__main__':
    tf.test.main()
