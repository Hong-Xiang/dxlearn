# _*_ encoding: utf-8 _*_

import tensorflow as tf 
import numpy as np

from dxl.learn.model.cnn import UnitBlock
from dxl.learn.model.cnn import ResidualIncept, ResidualStackedConv
from dxl.learn.model.cnn import StackedResidualIncept, StackedResidualConv
from dxl.learn.model.cnn import Conv2D, StackedConv2D, InceptionBlock, UnitBlock


class ResudualTestUniBlok(tf.test.TestCase):
    def test_ResidualIncept(self):
        unitblock_ins = UnitBlock(name='UnitBlock_test')
        x = np.ones([1, 10, 10, 3], dtype="float32")
        ratio = 0.5
        y_ = x + ratio * x

        residualincept_ins = ResidualIncept(
            name='ResidualIncept_test',
            input_tensor=tf.constant(x),
            ratio=ratio,
            sub_block=unitblock_ins)
        y = residualincept_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y, y_)
    
    def test_ResidualStackedConv(self):
        unitblock_ins = UnitBlock(name='UnitBlock_test')
        x = np.ones([1, 10, 10, 3], dtype="float32")
        ratio = 0.5
        y_ = x + ratio * x

        residualstackedconv_ins = ResidualStackedConv(
            name='ResidualStackedConv_test',
            input_tensor=tf.constant(x),
            ratio=ratio,
            sub_block=unitblock_ins)
        y = residualstackedconv_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y, y_)

    def test_StackedResidualIncept(self):
        unitblock_ins = UnitBlock(name='UnitBlock_test')
        x = np.ones([1, 10, 10, 3], dtype="float32")
        nb_layers = 2
        def_ratio = 0.3
        # default ResidualIncept ratio=0.3
        y_ = x
        for i in range(nb_layers):
            y_ += (y_ + y_ * def_ratio)

        stackedResidualincept_ins = StackedResidualIncept(
            name='StackedResidualIncept_test',
            input_tensor=tf.constant(x),
            nb_layers=nb_layers,
            sub_block=unitblock_ins)
        y = stackedResidualincept_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y, y_)

    def test_StackedResidualConv(self):
        unitblock_ins = UnitBlock(name='UnitBlock_test')
        x = np.ones([1, 10, 10, 3], dtype="float32")
        nb_layers = 2
        def_ratio = 0.1
        # default ResidualIncept ratio=0.1
        y_ = x
        for i in range(nb_layers):
            y_ += (y_ + y_ * def_ratio)

        stackedresidualconv_ins = StackedResidualConv(
            name='StackedResidualConv_test',
            input_tensor=tf.constant(x),
            nb_layers=nb_layers,
            sub_block=unitblock_ins)
        y = stackedresidualconv_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y, y_)
    

class ResidualTestDefaultBlock(tf.test.TestCase):
    def test_ResidualInceptDef(self):
        x = np.ones([1, 10, 10, 3], dtype="float32")
        ratio = 0.5
        residualincept_ins = ResidualIncept(
            name='ResidualInceptDef_testd',
            input_tensor=tf.constant(x),
            ratio=ratio)
        y = residualincept_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, (1,10,10,3))

    def test_ResidualStackedConvDef(self):
        x = np.ones([1, 10, 10, 3], dtype="float32")
        ratio = 0.5
        residualstackedconv_ins = ResidualStackedConv(
            name='ResidualStackedConvDef_test',
            input_tensor=tf.constant(x),
            ratio=ratio)
        y = residualstackedconv_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, (1,10,10,3))

    def test_StackedResidualInceptDef(self):
        x = np.ones([1, 10, 10, 3], dtype="float32")
        nb_layers = 2
        # default ResidualIncept ratio=0.3
        stackedResidualincept_ins = StackedResidualIncept(
            name='StackedResidualInceptDef_test',
            input_tensor=tf.constant(x),
            nb_layers=nb_layers)
        y = stackedResidualincept_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, (1,10,10,3))

    def test_StackedResidualConvDef(self):
        x = np.ones([1, 10, 10, 3], dtype="float32")
        nb_layers = 2
        # default ResidualIncept ratio=0.1
        stackedresidualconv_ins = StackedResidualConv(
            name='StackedResidualConvDef_test',
            input_tensor=tf.constant(x),
            nb_layers=nb_layers)
        y = stackedresidualconv_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, (1,10,10,3))


class ResidualTestInputBlock(tf.test.TestCase):
    def test_ResidualInceptInp(self):
        x = np.ones([1, 10, 10, 3], dtype="float32")
        ratio = 0.5
        sub_block =  InceptionBlock(
            name='InceptionBlock_block',
            input_tensor=tf.constant(x),
            paths=3,
            activation='incept')
        residualincept_ins = ResidualIncept(
            name='ResidualInceptDef_testd',
            input_tensor=tf.constant(x),
            ratio=ratio,
            sub_block=sub_block)
        y = residualincept_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, (1,10,10,3))

    def test_ResidualStackedConvInp(self):
        x = np.ones([1, 10, 10, 3], dtype="float32")
        ratio = 0.5
        sub_block = StackedConv2D(
            name='StackedConv2D_block',
            input_tensor=tf.constant(x),
            nb_layers=4,
            filters=1,
            kernel_size=[1,1],
            strides=(1, 1),
            padding='same',
            activation='basic')
        residualstackedconv_ins = ResidualStackedConv(
            name='ResidualStackedConv_test',
            input_tensor=tf.constant(x),
            ratio=ratio,
            sub_block=sub_block)
        y = residualstackedconv_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, (1,10,10,3))

    def test_StackedResidualInceptInp(self):
        x = np.ones([1, 10, 10, 3], dtype="float32")
        nb_layers = 2
        ratio = 0.5
        # default ResidualIncept ratio=0.3
        sub_block = ResidualStackedConv(
            name='ResidualStackedConv_block',
            input_tensor=tf.constant(x),
            ratio=ratio)
        stackedResidualincept_ins = StackedResidualIncept(
            name='StackedResidualInceptDef_test',
            input_tensor=tf.constant(x),
            nb_layers=nb_layers)
        y = stackedResidualincept_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, (1,10,10,3))

    def test_StackedResidualConvInp(self):
        x = np.ones([1, 10, 10, 3], dtype="float32")
        nb_layers = 2
        ratio = 0.3
        # default ResidualIncept ratio=0.1
        sub_block = ResidualStackedConv(
            name='ResidualStackedConv_test',
            input_tensor=tf.constant(x),
            ratio=ratio)
        stackedresidualconv_ins = StackedResidualConv(
            name='StackedResidualConvDef_test',
            input_tensor=tf.constant(x),
            nb_layers=nb_layers,
            sub_block=sub_block)
        y = stackedresidualconv_ins.outputs['main']
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, (1,10,10,3))


if __name__ == "__main__":
    tf.test.main()