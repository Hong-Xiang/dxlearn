
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

op = tf.load_op_library(
    '/home/chengaoyu/tools/tensorflow/bazel-bin/tensorflow/core/user_ops/siddon_gpu.so')

class TensorImage:
    def projection(self):
        projection_value = op.projection_gpu(model="siddon",
                                      lors=tf.constant(
                                          [[50., 0., 50.,-50., 0., 0., 50.],], dtype=tf.float32),  # input LORs
                                      tof_bin=1e-15,  # TOF parameter: 1st element is the TOF bin(mostly lower than 30e-12 ps (I set it to 10e-12 ps)); 2nd element is TOF time resolution (unit: ps)  ###if we dont have TOF, the 2nd element must be 0.
                                      time_resolution = 1e-10,
                                      grid=[30, 30, 30],       # voxel numbers of image (X,Y,Z)
                                      origin=[-15., -15., -15.],    # the coordinate of image lower-left corner (mostly I set the center of image to be origin of coordinate, so this parameter would be negative half value of image size) 
                                      size=[1.] * 3,    # voxel size of image
                                      image=tf.constant(
                                          np.ones([30, 30, 30], dtype=np.float32)), # image need to be projected
                                           # attenuation map (same size of image) ### if we dont have attentuation map, all element of thie array must be 0.
                                      )

        print(projection_value.eval())


class TensorDP:

    def backprojection(self):
        back_image = op.backprojection_gpu(model="siddon",
                                       image=tf.constant(
                                           np.ones([5, 5, 5], dtype=np.float32)),
                                       lors=tf.constant(
                                          [[0., 0., 20., 0., 0., -20., 0],], dtype=tf.float32),
                                       lor_values=tf.constant(
                                           [1., ], dtype=tf.float32),
                                       tof_bin = 1e-15,
                                       time_resolution = 2,
                                       grid=[5, 5, 5],
                                       origin=[-15, -15, -15.],
                                       size=[6., 6., 6.],
                                    #    atan_map=tf.constant(
                                        #   np.zeros([30,30,30] ,dtype=np.float32)),
                                       )
       
        print(back_image.eval())
        # print(sum(sum(back_image.eval())))



if __name__ == "__main__":
    with tf.Session() as sess:
        t=TensorImage()
        t.projection()
        y=TensorDP()
        y.backprojection()

  
  

