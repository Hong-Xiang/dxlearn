from dxl.learn.core import Tensor
from dxl.learn.core import GraphInfo
import tensorflow as tf

class GlobalStep(Tensor):
    def __init__(self):
        super().__init__(tf.train.get_or_create_global_step(), GraphInfo())

    def increased(self):
        with tf.name_scope('global_step_increased'):
            return tf.assign_add(self.data, 1)
    
    def current_step(self):
        return self.data