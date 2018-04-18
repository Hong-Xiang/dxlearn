import tensorflow as tf
import numpy as np
import os
import warnings
import unittest
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')
op = tf.load_op_library(
    TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/tof_tor.so')
warnings.warn(DeprecationWarning())

projection = op.projection_gpu
backprojection = op.backprojection_gpu


class TofOPGridCase(unittest.TestCase):
    """
    grid image test for tof tor op
    """

    def setUp(self):
        self.model = "tor"
        self.kernel_width = 1
        # voxel numbers of image (X,Y,Z)
        self.grid = [10, 10, 10]
        self.position = [0., 0., 0.]
        self.size = [10., 10., 10.]

    def testProjection(self):
        """
        """
        self.image = tf.constant(
            np.ones([1, 1, 10], dtype=np.float32))
        # input LORs
        self.lors = tf.transpose(tf.constant(
            [[0.0, 0.0, -6.0,
              0.0, 0.0, 6.0,
              0.0, 0.0, 0.0], ], dtype=tf.float32))

        result = projection(
            lors=self.lors,
            image=self.image,
            grid=self.grid,
            center=self.position,
            size=self.size,
            kernel_width=self.kernel_width,
            model=self.model
        )
        # print(result.eval().shape)
        self.assertAlmostEqual(float(result.eval()), 4.215969, 5)

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
        self.assertAlmostEqual(float(np.sum(result)), 4.215969, 5)


if __name__ == "__main__":
    with tf.Session() as sess:
        unittest.main()
