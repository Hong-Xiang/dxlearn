import tensorflow as tf
from enum import Enum


class ActivationConfig:
    def __init__(self, name, pre_activation=False, post_activation=False):
        self.name = name
        self.pre_activation = pre_activation
        self.post_activation = post_activation


class ActivationConfigs(Enum):
    basic = ActivationConfig('relu', False, True)
    linear = ActivationConfig('none')
    none = ActivationConfig('none')
    res_celu = ActivationConfig('celu', True, False)
    incept = ActivationConfig('celu', True, False)
    pre = ActivationConfig('celu', True, False)    


def celu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    with tf.name_scope('celu'):
        x = tf.nn.elu(tf.concat([x, -x], axis=-1))
    return x


def apply(activation_config, input_tensor, position=None):
    if position is None:
        with tf.name_scope('activation'):
            return get_activation(activation_config.name)(input_tensor)
    elif position.lower() == 'pre' and activation_config.pre_activation:
        with tf.name_scope('pre_activation'):
            return get_activation(activation_config.name)(input_tensor)
    elif position.lower() == 'post' and activation_config.post_activation:
        with tf.name_scope('post_activation'):
            return get_activation(activation_config.name)(input_tensor)
    return input_tensor


def unified_config(name):
    if isinstance(name, ActivationConfig):
        return name
    for e in ActivationConfigs:
        if e == name or e.name == name:
            return e.value
    return None


def get_activation(name):
    name = name.lower()
    activations = {
        'relu': tf.nn.relu,
        'elu': tf.nn.elu,
        'celu': celu,
        'crelu': tf.nn.crelu,
        'none': lambda x: x}
    alias = {}
    for n in alias:
        if name in alias[n]:
            name = n
    if name in activations:
        return activations[name]
    raise ValueError("Unknown name of activation: {}.".format(name))
