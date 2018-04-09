import tensorflow as tf
import numpy as np
import os
import warnings
import unittest
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')
op = tf.load_op_library(
    TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/pet_gpu.so')
warnings.warn(DeprecationWarning())

projection = op.projection_gpu
backprojection = op.backprojection_gpu

class TorMeshTest(unittest.TestCase):
    """
    a single mesh image test for Tor op
    """
    def setUp(self):
        self.model = "tor"
        self.kernel_width = 1
        self.grid = [1, 1, 1]
        self.position = [0., 0., 0.]
        self.size = [1., 1., 1.]
        self.image = tf.constant(
            np.ones(self.grid, dtype=np.float32)
        )
    def testCenterPass(self):
        self.lors = tf.transpose(
            tf.constant([[0.0, 0.0, -10.,
                          0.0, 0.0,  10.], ], dtype=np.float32)
        )
        result = projection(lors=self.lors,
                            image=self.image,
                            grid=self.grid,
                            center=self.position,
                            size=self.size,
                            kernel_width=self.kernel_width,
                            model=self.model
                            )
        # print(result.eval())
        self.assertAlmostEqual(float(result.eval()), 1.0, 5)
    
    def testCenterBack(self):
        self.lors = tf.transpose(
            tf.constant([[0.0, 0.0, -10.,
                          0.0, 0.0,  10.], ], dtype=np.float32)
        )
        self.line_integral = tf.constant(
            [[1.], ], dtype=tf.float32)
        result = projection(lors=self.lors,
                            image=self.image,
                            grid=self.grid,
                            center=self.position,
                            size=self.size,
                            kernel_width=self.kernel_width,
                            model=self.model
                            )
        # print(result.eval())
        self.assertAlmostEqual(float(result.eval()), 1.0, 5)


    def testEdgePass(self):
        self.lors = tf.transpose(
            tf.constant([[0.0, 0.5, -10.,
                          0.0, 0.5,  10.], ], dtype=np.float32)
        )
        result = projection(lors=self.lors,
                            image=self.image,
                            grid=self.grid,
                            center=self.position,
                            size=self.size,
                            kernel_width=self.kernel_width,
                            model=self.model
                            )
        # print(result.eval())
        self.assertAlmostEqual(float(result.eval()), 0.01110899, 5)

    def testEdgeBack(self):
        self.lors = tf.transpose(
            tf.constant([[0.0, 0.5, -10.,
                          0.0, 0.5,  10.], ], dtype=np.float32)
        )
        self.line_integral = tf.constant(
            [[1.], ], dtype=tf.float32)
        result = projection(lors=self.lors,
                            image=self.image,
                            grid=self.grid,
                            center=self.position,
                            size=self.size,
                            kernel_width=self.kernel_width,
                            model=self.model
                            )
        # print(result.eval())
        self.assertAlmostEqual(float(result.eval()), 0.01110899, 5)
    def testVertexPass(self):
        self.lors = tf.transpose(
            tf.constant([[-0.5, 0.5, -10.,
                          -0.5, 0.5,  10.], ], dtype=np.float32)
        )
        result = projection(lors=self.lors,
                            image=self.image,
                            grid=self.grid,
                            center=self.position,
                            size=self.size,
                            kernel_width=self.kernel_width,
                            model=self.model
                            )
        # print(result.eval())
        self.assertAlmostEqual(float(result.eval()), 0.0, 5)

    def testVertexBack(self):
        self.lors = tf.transpose(
            tf.constant([[-0.5, 0.5, -10.,
                          -0.5, 0.5,  10.], ], dtype=np.float32)
        )
        self.line_integral = tf.constant(
            [[1.], ], dtype=tf.float32)
        result = projection(lors=self.lors,
                            image=self.image,
                            grid=self.grid,
                            center=self.position,
                            size=self.size,
                            kernel_width=self.kernel_width,
                            model=self.model
                            )
        # print(result.eval())
        self.assertAlmostEqual(float(result.eval()), 0.0, 5)

class TorOPSliceTest(unittest.TestCase):
    """
    a single slice image test for Tor op
    """

    def setUp(self):
        self.model = "tor"
        self.kernel_width = 2
        self.grid = [4, 4, 1]
        self.position = [0., 0., 0.]
        self.size = [4., 4., 1.]

    def testProjection(self):
        self.image = tf.constant(
            np.ones(self.grid, dtype=np.float32)
        )
        self.lors = tf.transpose(
            tf.constant([[0.0, 0.0, -10.,
                          0.0, 0.0,  10.], ], dtype=np.float32)
        )
        result = projection(lors=self.lors,
                            image=self.image,
                            grid=self.grid,
                            center=self.position,
                            size=self.size,
                            kernel_width=self.kernel_width,
                            model=self.model
                            )
        # print(result.eval())
        self.assertAlmostEqual(float(result.eval()), 0.4215969, 5)

    # def testObliqueProjection(self):
    #     """
    #     an oblique lor pass through a 
    #     """
    #     self.image = tf.constant(
    #         np.ones(self.grid, dtype=np.float32)
    #     )
    #     self.lors = tf.transpose(
    #         tf.constant([[0.0, 0.0, -0.5,
    #                       0.2, 0.2,  0.5], ], dtype=np.float32)
    #     )
    #     result = projection(lors=self.lors,
    #                         image=self.image,
    #                         grid=self.grid,
    #                         center=self.position,
    #                         size=self.size,
    #                         kernel_width=self.kernel_width,
    #                         model=self.model
    #                         )
    #     # print(result.eval())
    #     self.assertAlmostEqual(float(result.eval()), 0.54094778, 5)



    def testBackprojection(self):
        self.image = tf.constant(
            np.ones(self.grid, dtype=np.float32))
        self.lors = tf.transpose(
            tf.constant([[0., 0., -20.,
                          0., 0.,  20.], ], dtype=tf.float32))
        self.line_integral = tf.constant(
            [[1.], ], dtype=tf.float32)
        result = backprojection(lors=self.lors,
                                image=self.image,
                                grid=self.grid,
                                center=self.position,
                                size=self.size,
                                line_integral=self.line_integral,
                                kernel_width=self.kernel_width,
                                model=self.model
                                )
        # assert result.eval() == []

class TorOPGridCase(unittest.TestCase):
    """
    grid image test for Tor op
    """

    def setUp(self):
        self.model = "tor"
        self.kernel_width = 2
        # voxel numbers of image (X,Y,Z)
        self.grid = [10, 10, 10]
        self.position = [0., 0., 0.]
        self.size = [10., 10., 10.]

    def testProjection(self):
        self.image = tf.constant(
            np.ones(self.grid, dtype=np.float32))  # image need to be projected
        self.lors = tf.transpose(
            tf.constant([[0., 0., -20.,
                          0., 0.,  20.], ], dtype=tf.float32))
        result = projection(lors=self.lors,
                            image=self.image,
                            grid=self.grid,
                            center=self.position,
                            size=self.size,
                            kernel_width=self.kernel_width,
                            model=self.model
                            )
        # print(result.eval().shape)
        self.assertAlmostEqual( float(result.eval()), 4.215969, 5)

    def testBackprojection(self):
        self.image = tf.constant(
            np.ones(self.grid, dtype=np.float32))
        self.lors = tf.transpose(
            tf.constant([[0., 0., -20.,
                          0., 0.,  20.], ], dtype=tf.float32))
        self.line_integral = tf.constant(
            [[1.], ], dtype=tf.float32)
        result = backprojection(lors=self.lors,
                                image=self.image,
                                grid=self.grid,
                                center=self.position,
                                size=self.size,
                                line_integral=self.line_integral,
                                kernel_width=self.kernel_width,
                                model=self.model
                                )

        # print(result.eval().shape)
        result = np.array(result.eval()).reshape((-1))
        self.assertAlmostEqual( float(np.sum(result)), 4.215969, 5)


# class TorOPTestSuite(unittest.TestSuite):
#     """

#     """

#     def __init__(self):
#         unittest.TestSuite.__init__(self, map(TorOPGridCase,
#                                               ("testProjection",
#                                                "testBackprojection")))


if __name__ == "__main__":
    with tf.Session() as sess:
        unittest.main()
