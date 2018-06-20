# _*_ encoding: utf-8 _*_

import tensorflow as tf
import numpy as np
from dxl.learn.test import TestCase
from dxl.learn.model.cnn import Conv2D, InceptionBlock, DownSampling2D, UpSampling2D
from dxl.learn.test import UnitBlock
from dxl.learn.model.stack import  StackedConv2D
import pytest


class TestConv2D(TestCase):
    def get_input(self):
        return tf.constant(np.ones([1, 100, 100, 3], dtype="float32"))

    def make_model(self, name=None):
        if name is None:
            name = 'conv2d_test'
        return Conv2D(
            name,
            inputs=self.get_input(),
            filters=32,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding='same',
            activation='basic')

    def expected_output_shape(self):
        return (1, 50, 50, 32)

    def test_shape(self):
        conv2d_ins = self.make_model()
        y = conv2d_ins()
        # y = conv2d_ins()
        with self.variables_initialized_test_session() as sess:
            y = sess.run(y)
            self.assertAllEqual(y.shape, self.expected_output_shape())

    def test_scope(self):
        m0 = self.make_model('conv0')
        m1 = self.make_model('conv1')

    def test_scope_fail(self):
        with pytest.raises(ValueError):
            m0 = self.make_model('conv0')
            m0.make()
            m1 = self.make_model('conv0')
            m1.make()

    def test_scope2(self):
        m2 = self.make_model('scope/conv0')
        m3 = self.make_model('scope/conv1')

    def test_reuse(self):
        m = self.make_model()
        xt = m({'input': self.get_input()})
        with self.variables_initialized_test_session() as sess:
            yt = sess.run(xt)
            self.assertAllEqual(yt.shape, self.expected_output_shape())


class TestBlocks(TestCase):
    def get_input(self):
        return tf.constant(np.ones([2, 100, 100, 3], dtype="float32"))

    def test_InceptionBlock(self):
        inceptionblock_ins = InceptionBlock(
            'InceptionBlock_test',
            inputs=self.get_input(),
            paths=3,
            activation='incept')
        y = inceptionblock_ins()
        self.assertAllEqual(y.shape, (2, 100, 100, 3))

    def test_DownSampling2D_Def(self):
        downsampling2d_ins = DownSampling2D(
            "DownSampling2D_test",
            inputs=self.get_input(),
            size=(0.5, 0.5))
        y = downsampling2d_ins()
        self.assertAllEqual(y.shape, (2, 50, 50, 3))

    def test_DownSampling2D_Inp(self):
        downsampling2d_ins = DownSampling2D(
            "DownSampling2D_test",
            inputs=self.get_input(),
            size=(30, 30),
            is_scale=False,
            method=2)
        y = downsampling2d_ins()
        self.assertAllEqual(y.shape, (2, 30, 30, 3))

    def test_UpSampling2D_Def(self):
        upsampling2d_ins = UpSampling2D(
            "DownSampling2D_test",
            inputs=self.get_input(),
            size=(1.5, 1.5))
        y = upsampling2d_ins()
        self.assertAllEqual(y.shape, (2, 150, 150, 3))

    def test_UpSampling2D_Inp(self):
        upsampling2d_ins = UpSampling2D(
            "DownSampling2D_test",
            inputs=self.get_input(),
            size=(130, 130),
            is_scale=False,
            method=3)
        y = upsampling2d_ins()
        self.assertAllEqual(y.shape, (2, 130, 130, 3))


if __name__ == "__main__":
    tf.test.main()
