import tensorflow as tf 
import numpy as np
from dxl.learn.model.image import random_crop, boundary_crop, align_crop
from dxl.learn.model.image import random_crop_offset

class CropTest(tf.test.TestCase):
    def test_random_crop_offset(self):
        input_shape = [32, 512, 512, 3]
        target_shape = [32, 224, 244, 3]
        # test batched = False
        offset = random_crop_offset(input_shape, target_shape,)
        comp0 = [True, True, True, True]
        self.assertAllEqual(list(map(lambda x: x>=0, offset)), comp0)

    def test_random_crop(self):
        x = tf.constant([[[1, 1, 1], [2, 2, 2]],
                         [[3, 3, 3], [4, 4, 4]],
                         [[5, 5, 5], [6, 6, 6]],
                         [[7, 7, 7], [8, 8, 8]]])
        target_shape = [1, 1, 3]
        y = random_crop(
            input_=x,
            target_shape=target_shape,
            name='random_crop'
        )
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, target_shape)

    def test_align_crop(self):
        x = np.ones([32, 100, 100, 3], dtype="float32")
        target_shape = [32, 20, 20, 3]
        # test offset=None
        y = align_crop(x, target_shape)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, target_shape)
        # test input offset
        offset = [0, 50, 50, 0]
        y = align_crop(x, target_shape, offset)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, target_shape)

    def test_boundary_crop(self):
        x = np.ones([32, 100, 100, 3], dtype="float32")
        offset = [0, 20, 20, 0]
        shape_y = [s - 2 * o for s, o in zip(x.shape, offset)]

        y = boundary_crop(x, offset)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            y = sess.run(y)
            self.assertAllEqual(y.shape, shape_y)


if __name__ == "__main__":
    tf.test.main()