from dxpy.configs import configurable
import tensorflow as tf
from ..config import config
from ..model.tensor import ReduceSum


@configurable(config, with_name=True)
def poission_noise_simulator(tensor, summation_value, name='poission_noise_simulator')
    tensor = ReduceSum('{}/reduce_sum'.format(name),
                       tensor,
                       fixed_summation_value=summation_value).as_tensor()
    with tf.name_scope('add_poission_noise'):
        tensor = tf.random_poisson(tensor, shape=[])
    return tensor
