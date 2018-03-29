import tensorflow as tf
import numpy as np
op = tf.load_op_library(
    '/home/chengaoyu/tools/tensorflow/bazel-bin/tensorflow/core/user_ops/pet_gpu.so')
# sop = tf.load_op_library(
    # '/home/chengaoyu/tools/tensorflow/bazel-bin/tensorflow/core/user_ops/siddon_gpu.so')

# op = sop

# sp  = sop.projection_gpu
# sbp = sop.backprojection_gpu


projection = op.projection_gpu
backprojection = op.backprojection_gpu


def computeMap(grid, center, size, xlors, ylors, zlors):
    # imgz = inputs[self.KEYS.TENSOR.IMAGE].data
    # grid = grid[::-1]
    # size = size[::-1]
    # center = center[::-1]
    imgz = tf.zeros(grid[::-1], tf.float32)
    imgx = tf.transpose(imgz, perm=[2, 0, 1])
    imgy = tf.transpose(imgz, perm=[1, 0, 2])
    xlors0 = tf.constant(xlors, tf.float32)
    ylors0 = tf.constant(ylors, tf.float32)
    zlors0 = tf.constant(zlors, tf.float32)
    # grid = inputs[self.KEYS.TENSOR.GRID].data
    # center = inputs[self.KEYS.TENSOR.CENTER].data
    # size = inputs[self.KEYS.TENSOR.SIZE].data
    # xlors = inputs[self.KEYS.TENSOR.LORS_X].data
    # ylors = inputs[self.KEYS.TENSOR.LORS_Y].data
    # zlors = inputs[self.KEYS.TENSOR.LORS_Z].data

    # lors tranposed
    xlors = tf.transpose(xlors0)
    ylors = tf.transpose(ylors0)
    zlors = tf.transpose(zlors0)

    xproj = tf.ones(xlors.shape.as_list()[1], 1)
    yproj = tf.ones(ylors.shape.as_list()[1], 1)
    zproj = tf.ones(zlors.shape.as_list()[1], 1)

    model = 'tor'
<<<<<<< HEAD
    # kernel_width = np.sqrt(6.8 * 6.8 / np.pi)
    kernel_width = np.sqrt(3.4 * 3.4 / np.pi)
=======
    kernel_width = np.sqrt(3.4 * 3.4 / np.pi)
    # kernel_width = np.sqrt(20 * 20 / np.pi)
>>>>>>> master

    bpz = backprojection(image=imgz, grid=grid, lors=zlors,
                         center=center, size=size, line_integral=zproj,  kernel_width=kernel_width, model=model)

    gridx = tf.constant(
        np.array([grid[1], grid[2], grid[0]], dtype=np.int32), name='gridx')
    centerx = tf.constant(
        np.array([center[1], center[2], center[0]], np.float32), name='centerx')
    sizex = tf.constant(
        np.array([size[1], size[2], size[0]], np.float32), name='sizex')

    # gridx = inputs[self.KEYS.TENSOR.GRID].data
    # centerx = inputs[self.KEYS.TENSOR.CENTER].data
    # sizex = inputs[self.KEYS.TENSOR.SIZE].data

    bpx = backprojection(image=imgx, grid=gridx, lors=xlors,
                         center=centerx, size=sizex, line_integral=xproj,  kernel_width=kernel_width, model=model)
    bpxt = tf.transpose(bpx, perm=[1, 2, 0])

    gridy = tf.constant(
        np.array([grid[0], grid[2], grid[1]], np.int32), name='gridy')
    centery = tf.constant(
        np.array([center[0], center[2], center[1]], np.float32), name='centery')
    sizey = tf.constant(
        np.array([size[0], size[2], size[1]], np.float32), name='sizey')

    # gridy = inputs[self.KEYS.TENSOR.GRID].data
    # centery = inputs[self.KEYS.TENSOR.CENTER].data
    # sizey = inputs[self.KEYS.TENSOR.SIZE].data

    bpy = backprojection(image=imgy, grid=gridy, lors=ylors,
                         center=centery, size=sizey, line_integral=yproj,  kernel_width=kernel_width, model=model)
    bpyt = tf.transpose(bpy, perm=[1, 0, 2])

    result = bpxt + bpyt + bpz
    # result = bpxt + bpyt
    result = tf.transpose(result)
    config = tf.ConfigProto()    
    config.gpu_options.allow_growth = True    
    with tf.Session(config=config) as sess:
        result = sess.run(result)
    tf.reset_default_graph()
    return result
    # result = imgz / (effmap+1e-8) * bpz
    # return Tensor(result, None, self.graph_info.update(name=None))


def siddonMap(grid, size, origin, lors):
    pass
    # grid = grid[::-1]
    # origin = origin[::-1]
    # size = size[::-1]
    # lors = tf.constant(lors, tf.float32)
    # proj = tf.ones(lors.shape.as_list()[0], 1)
    # img = tf.zeros(grid, tf.float32)
    # result = sbp(model = 'siddon', image=img, grid=grid, origin=origin, size=size,
    #                         lors=lors, lor_values=proj,
    #                         tof_bin = 1e-15, time_resolution=2)
    # # result = tf.transpose(result)

    # with tf.Session() as sess:
    #     return sess.run(result)

