from dxl.learn.core import Model, Tensor
# from dxl.learn.model.tor_recon import Projection, BackProjection
import tensorflow as tf
import numpy as np
import warnings
import os
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')
op = tf.load_op_library(
    TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/siddon_gpu.so')
warnings.warn(DeprecationWarning())

projection = op.projection_gpu
backprojection = op.backprojection_gpu


class SiddonStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            # PROJECTION = 'projection'
            # SYSTEM_MATRIX = 'system_matrix'
            EFFICIENCY_MAP = 'efficiency_map'
            LORS = 'lors'

    def __init__(self, name, image, efficiency_map,
                 grid, origin, voxsize, time_res,
                 tof_bin, lors, graph_info):
        self.grid = grid
        self.origin = origin
        self.voxsize = voxsize
        self.time_res = time_res
        self.tof_bin = tof_bin
        super().__init__(
            name,
            {
                self.KEYS.TENSOR.IMAGE:
                image,
                self.KEYS.TENSOR.EFFICIENCY_MAP:
                efficiency_map,
                self.KEYS.TENSOR.LORS:
                lors
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        # the default order of the image is z-dominant(z,y,x)
        # for projection another two images are created.
        image = inputs[self.KEYS.TENSOR.IMAGE].data
        effmap = inputs[self.KEYS.TENSOR.EFFICIENCY_MAP].data

        grid = self.grid
        origin = self.origin
        size = self.voxsize
        lors = inputs[self.KEYS.TENSOR.LORS].data
        model = 'siddon'

        """
        to be check here:
        why the float value become a tuble?
        """
        tof_bin = self.tof_bin[0]
        time_res = self.time_res[0]

        
        # z-dominant, no transpose        
        print("this is the time_res:",time_res)        
        p = projection(
            lors=lors,
            image=image,
            grid=grid,
            origin=origin,
            size=size,
            time_resolution = time_res,
            tof_bin = tof_bin,
            model = model)

        bp = backprojection(
            image=image,
            grid=grid,
            origin=origin,
            size=size,
            lors=lors,
            lor_values=p,
            tof_bin = tof_bin,
            time_resolution = time_res,
            model=model)

        result = image / (effmap + 1e-8) * bp
        return Tensor(result, None, self.graph_info.update(name=None))
