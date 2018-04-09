from dxl.learn.core import Model, Tensor
# from dxl.learn.model.tor_recon import Projection, BackProjection
import tensorflow as tf
import numpy as np
import warnings
import os
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')
op = tf.load_op_library(
    TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/tof_tor.so')
warnings.warn(DeprecationWarning())

projection = op.projection_gpu
backprojection = op.backprojection_gpu


class TorStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            # PROJECTION = 'projection'
            # SYSTEM_MATRIX = 'system_matrix'
            EFFICIENCY_MAP = 'efficiency_map'
            LORS_X = 'xlors'
            LORS_Y = 'ylors'
            LORS_Z = 'zlors'

    def __init__(self, name, image, efficiency_map,
                 grid, position, size,
                 kernel_width, tof_bin, tof_sigma2,
                 xlors, ylors, zlors,
                 graph_info):
        self.grid = np.array(grid, dtype=np.int32)
        self.position = np.array(position, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.kernel_with = float(kernel_width)
        self.tof_bin = float(tof_bin)
        self.tof_sigma2 = float(tof_sigma2)
        super().__init__(
            name,
            {
                self.KEYS.TENSOR.IMAGE:
                image,
                self.KEYS.TENSOR.EFFICIENCY_MAP:
                efficiency_map,
                self.KEYS.TENSOR.LORS_X:
                xlors,
                self.KEYS.TENSOR.LORS_Y:
                ylors,
                self.KEYS.TENSOR.LORS_Z:
                zlors
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        # the default order of the image is z-dominant(z,y,x)
        # for projection another two images are created.
        imgz = inputs[self.KEYS.TENSOR.IMAGE].data

        imgx = tf.transpose(imgz, perm=[2, 0, 1])
        imgy = tf.transpose(imgz, perm=[1, 0, 2])

        effmap = inputs[self.KEYS.TENSOR.EFFICIENCY_MAP].data

        model = 'tor'
        grid = self.grid
        position = self.position
        size = self.size
        kernel_width = self.kernel_with
        tof_bin = self.tof_bin
        tof_sigma2 = self.tof_sigma2
        # print('tof_bin:', tof_bin)

        xlors = inputs[self.KEYS.TENSOR.LORS_X].data
        ylors = inputs[self.KEYS.TENSOR.LORS_Y].data
        zlors = inputs[self.KEYS.TENSOR.LORS_Z].data

        # lors tranposed
        xlors = tf.transpose(xlors)
        ylors = tf.transpose(ylors)
        zlors = tf.transpose(zlors)

        # z-dominant, no transpose
        pz = projection(
            lors=zlors,
            image=imgz,
            grid=grid,
            position=position,
            size=size,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2,)

        bpz = backprojection(
            image=imgz,
            grid=grid,
            lors=zlors,
            position=position,
            size=size,
            lor_values=pz,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2,
        )
        # x-dominant, tranposed
        gridx = tf.constant(
            np.array([grid[0], grid[2], grid[1]]), name='gridx')
        positionx = tf.constant(
            np.array([position[0], position[2], position[1]]), name='centerx')
        sizex = tf.constant(
            np.array([size[0], size[2], size[1]]), name='sizex')
        px = projection(
            lors=xlors,
            image=imgx,
            grid=gridx,
            position=positionx,
            size=sizex,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2,
        )

        bpx = backprojection(
            image=imgx,
            grid=gridx,
            lors=xlors,
            position=positionx,
            size=sizex,
            lor_values=px,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2,)
        bpxt = tf.transpose(bpx, perm=[1, 2, 0])

        # y-dominant, tranposed
        # gridy = grid
        # centery = position
        # sizey = size
        gridy = tf.constant(
            np.array([grid[1], grid[2], grid[0]]), name='gridy')
        positiony = tf.constant(
            np.array([position[1], position[2], position[0]]), name='centery')
        sizey = tf.constant(
            np.array([size[1], size[2], size[0]]), name='sizey')
        py = projection(
            lors=ylors,
            image=imgy,
            grid=gridy,
            position=positiony,
            size=sizey,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2,)

        bpy = backprojection(
            image=imgy,
            grid=gridy,
            lors=ylors,
            position=positiony,
            size=sizey,
            lor_values=py,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2,)
        bpyt = tf.transpose(bpy, perm=[1, 0, 2])

        result = imgz * effmap * (bpxt + bpyt + bpz)
        # result = imgz / (effmap+1e-8) * bpz
        return Tensor(result, None, self.graph_info.update(name=None))


class Projection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            # PROJECTION = 'projection'
            # SYSTEM_MATRIX = 'system_matrix'
            # EFFICIENCY_MAP = 'efficiency_map'
            LORS = 'lors'

    def __init__(self, name, image,
                 grid, position, size,
                 lors,
                 tof_bin, tof_sigma2,
                 kernel_width,
                 graph_info):
        self.grid = np.array(grid, dtype=np.int32)
        self.position = np.array(position, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.kernel_width = float(kernel_width)
        self.tof_bin = float(tof_bin)
        self.tof_sigma2 = float(tof_sigma2)
        print(tof_bin)
        super().__init__(
            name,
            {
                self.KEYS.TENSOR.IMAGE: image,
                self.KEYS.TENSOR.LORS: lors
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        img = inputs[self.KEYS.TENSOR.IMAGE].data
        grid = self.grid
        position = self.position
        size = self.size
        lors = inputs[self.KEYS.TENSOR.LORS].data
        lors = tf.transpose(lors)
        kernel_width = self.kernel_width
        tof_bin = self.tof_bin
        tof_sigma2 = self.tof_sigma2
        projection_value = projection(
            lors=lors,
            image=img,
            grid=grid,
            position=position,
            size=size,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2)
        return Tensor(projection_value, None, self.graph_info.update(name=None))


class BackProjection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            # PROJECTION = 'projection'
            # SYSTEM_MATRIX = 'system_matrix'
            # EFFICIENCY_MAP = 'efficiency_map'
            LORS = 'lors'

    def __init__(self, name, image,
                 grid, position, size,
                 lors,
                 tof_bin, tof_sigma2,
                 kernel_width,
                 graph_info):
        self.grid = np.array(grid, dtype=np.int32)
        self.position = np.array(position, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.kernel_width = float(kernel_width)
        self.tof_bin = float(tof_bin)
        self.tof_sigma2 = float(tof_sigma2)
        print(tof_bin)
        super().__init__(
            name,
            {
                self.KEYS.TENSOR.IMAGE: image,
                self.KEYS.TENSOR.LORS: lors
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        img = inputs[self.KEYS.TENSOR.IMAGE].data
        grid = self.grid
        position = self.position
        size = self.size
        lors = inputs[self.KEYS.TENSOR.LORS].data
        lors = tf.transpose(lors)
        kernel_width = self.kernel_width
        tof_bin = self.tof_bin
        tof_sigma2 = self.tof_sigma2
        backprojection_image = backprojection(
            lors=lors,
            image=img,
            grid=grid,
            position=position,
            size=size,
            kernel_width=kernel_width,
            model=model,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2)
        return Tensor(backprojection_image, None, self.graph_info.update(name=None))
