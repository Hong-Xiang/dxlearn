# # _*_ encoding: utf-8 _*_

# import tensorflow as tf
# import numpy as np

# from dxl.learn.model.cnn import UnitBlock
# from dxl.learn.model.cnn import ResidualIncept, ResidualStackedConv
# from dxl.learn.model.cnn import StackedResidualIncept, StackedResidualConv
# from dxl.learn.model.cnn import Conv2D, StackedConv2D, InceptionBlock, UnitBlock

# import pytest


# class ResudualTestUniBlok(tf.test.TestCase):
#     def get_input(self):
#         return np.ones([1, 10, 10, 3], dtype="float32")

#     def get_model(self):
#         return UnitBlock("UnitBlock_test")

#     def test_ResidualIncept(self):
#         x = self.get_input()
#         ratio = 0.5
#         y_ = x + ratio * x

#         residualincept_ins = ResidualIncept(
#             'ResidualIncept_test',
#             inputs=tf.constant(x),
#             ratio=ratio,
#             graph=self.get_model())
#         y = residualincept_ins()
#         with self.test_session() as sess:
#             sess.run(tf.global_variables_initializer())
#             y = sess.run(y)
#             self.assertAllEqual(y, y_)

#     def test_ResidualStackedConv(self):
#         x = self.get_input()
#         ratio = 0.5
#         y_ = x + ratio * x

#         residualstackedconv_ins = ResidualStackedConv(
#             'ResidualStackedConv_test',
#             inputs=tf.constant(x),
#             ratio=ratio,
#             graph=self.get_model())
#         y = residualstackedconv_ins()
#         with self.test_session() as sess:
#             sess.run(tf.global_variables_initializer())
#             y = sess.run(y)
#             self.assertAllEqual(y, y_)

#     def test_StackedResidualIncept(self):
#         x = self.get_input()
#         nb_layers = 2
#         y_ = x
        
#         stackedResidualincept_ins = StackedResidualIncept(
#             'StackedResidualIncept_test',
#             inputs=tf.constant(x),
#             nb_layers=nb_layers,
#             graph=self.get_model())
#         y = stackedResidualincept_ins()
#         with self.test_session() as sess:
#             sess.run(tf.global_variables_initializer())
#             y = sess.run(y)
#             self.assertAllEqual(y, y_)

#     def test_StackedResidualConv(self):
#         x = self.get_input()
#         nb_layers = 2
#         y_ = x

#         stackedresidualconv_ins = StackedResidualConv(
#             'StackedResidualConv_test',
#             inputs=tf.constant(x),
#             nb_layers=nb_layers,
#             graph=self.get_model())
#         y = stackedresidualconv_ins()
#         with self.test_session() as sess:
#             sess.run(tf.global_variables_initializer())
#             y = sess.run(y)
#             self.assertAllEqual(y, y_)


# class ResidualTestDefaultBlock(tf.test.TestCase):
#     def get_input(self):
#         return np.ones([1, 10, 10, 3], dtype="float32")

#     def test_ResidualInceptDef(self):
#         x = self.get_input()
#         ratio = 0.5
#         residualincept_ins = ResidualIncept(
#             'ResidualInceptDef_test',inputs=tf.constant(x), ratio=ratio)
#         y = residualincept_ins()
#         self.assertAllEqual(y.shape, (1, 10, 10, 3))

#     def test_ResidualStackedConvDef(self):
#         x = self.get_input()
#         ratio = 0.5
#         residualstackedconv_ins = ResidualStackedConv(
#             'ResidualStackedConvDef_test',
#             inputs=tf.constant(x),
#             ratio=ratio)
#         y = residualstackedconv_ins()
#         self.assertAllEqual(y.shape, (1, 10, 10, 3))

#     def test_StackedResidualInceptDef(self):
#         x = self.get_input()
#         nb_layers = 2
#         # default ResidualIncept ratio=0.3
#         stackedResidualincept_ins = StackedResidualIncept(
#             'StackedResidualInceptDef_test',
#             inputs=tf.constant(x),
#             nb_layers=nb_layers)
#         y = stackedResidualincept_ins()
#         self.assertAllEqual(y.shape, (1, 10, 10, 3))

#     def test_StackedResidualConvDef(self):
#         x = self.get_input()
#         nb_layers = 2
#         # default ResidualIncept ratio=0.1
#         stackedresidualconv_ins = StackedResidualConv(
#             'StackedResidualConvDef_test',
#             inputs=tf.constant(x),
#             nb_layers=nb_layers)
#         y = stackedresidualconv_ins()
#         self.assertAllEqual(y.shape, (1, 10, 10, 3))


# class ResidualTestInputBlock(tf.test.TestCase):
#     def get_input(self):
#         return np.ones([1, 10, 10, 3], dtype="float32")

#     def test_ResidualInceptInp(self):
#         x = self.get_input()
#         ratio = 0.5
#         graph = InceptionBlock(
#             'InceptionBlock_block',
#             inputs=tf.constant(x),
#             paths=3,
#             activation='incept')
#         residualincept_ins = ResidualIncept(
#             'ResidualInceptInp_testd',
#             inputs=tf.constant(x),
#             ratio=ratio,
#             graph=graph)
#         y = residualincept_ins()
#         self.assertAllEqual(y.shape, (1, 10, 10, 3))

#     def test_ResidualStackedConvInp(self):
#         x = self.get_input()
#         ratio = 0.5
#         graph = StackedConv2D(
#             'StackedConv2D_block',
#             inputs=tf.constant(x),
#             nb_layers=4,
#             filters=1,
#             kernel_size=[1, 1],
#             strides=(1, 1),
#             padding='same',
#             activation='basic')
#         residualstackedconv_ins = ResidualStackedConv(
#             'ResidualStackedConv_test',
#             inputs=tf.constant(x),
#             ratio=ratio,
#             graph=graph)
#         y = residualstackedconv_ins()
#         self.assertAllEqual(y.shape, (1, 10, 10, 3))

#     def test_StackedResidualInceptInp(self):
#         x = self.get_input()
#         nb_layers = 2
#         ratio = 0.5
#         # default ResidualIncept ratio=0.3
#         graph = ResidualIncept(
#             'ResidualInput_block', inputs=tf.constant(x), ratio=ratio)
#         stackedResidualincept_ins = StackedResidualIncept(
#             'StackedResidualInceptInp_test',
#             inputs=tf.constant(x),
#             nb_layers=nb_layers,
#             graph=graph)
#         y = stackedResidualincept_ins()
#         self.assertAllEqual(y.shape, (1, 10, 10, 3))

#     def test_StackedResidualConvInp(self):
#         x = self.get_input()
#         nb_layers = 2
#         ratio = 0.3
#         # default ResidualIncept ratio=0.1
#         graph = ResidualStackedConv(
#             'ResidualStackedConv_test',
#             inputs=tf.constant(x),
#             ratio=ratio)
#         stackedresidualconv_ins = StackedResidualConv(
#             'StackedResidualConvDef_test',
#             inputs=tf.constant(x),
#             nb_layers=nb_layers,
#             graph=graph)
#         y = stackedresidualconv_ins()
#         self.assertAllEqual(y.shape, (1, 10, 10, 3))


# if __name__ == "__main__":
#     tf.test.main()