import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ops_root = '/home/hongxwing/Downloads/tensorflow/bazel-bin/tensorflow/core/user_ops/'
op = tf.load_op_library(ops_root + 'siddon_gpu.so')


class TensorImage:
  def projection(self):
    lors = tf.constant(
        [
            [50., 1., 2., -51., -1., -2., 0.],
            [60., 3., 4., -61., -3., -4., 0.],
        ],
        dtype=tf.float32)
    image = tf.constant(np.ones([30, 30, 1], dtype=np.float32))
    projection_value = op.projection_gpu(
        model="siddon",
        lors=lors,
        time_resolution=10e-12,  # unit: ps, set to zero if no TOF 
        tof_bin=200e-12,  # unit :ps, mostly lower than 30e-12 ps  
        grid=[30, 30, 1],  # voxel numbers of image (X,Y,Z)
        center=[0., 0., 0.],  # center of image
        size=[1., 1., 1.],  # voxel size 
        image=image,  # image need to be projected
    )
    print(projection_value.eval())


# class TensorDP:

#     def backprojection(self):
#         back_image = op.backprojection(model="siddon",
#                                        image=tf.constant(
#                                            np.ones([30, 30, 1], dtype=np.float32)),
#                                        lors=tf.constant(
#                                           [[3., 0., 0., -3., 0., 0., 0],], dtype=tf.float32),
#                                        lor_values=tf.constant(
#                                            [1., ], dtype=tf.float32),
#                                        tofinfo=tf.constant([10e-12,200e-12],dtype=tf.float32),
#                                        grid=tf.constant(
#                                            [30, 30, 1], dtype=tf.int32),
#                                        orgin=tf.constant(
#                                            [-15, -15, 0.], dtype=tf.float32),
#                                        size=tf.constant(
#                                            [1., 1., 1.], dtype=tf.float32),
#                                        atan_map=tf.constant(
#                                           np.zeros([30,30,1] ,dtype=np.float32)),
#                                        )

#         print(back_image.eval())

if __name__ == "__main__":
  with tf.Session() as sess:
    t = TensorImage()
    t.projection()
    # y=TensorDP()
    # y.backprojection()
