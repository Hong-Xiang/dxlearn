import tensorflow as tf
from ..core import Model
import warnings


def mean_square_error(label, data):
    with tf.name_scope('mean_squared_error'):
        return tf.sqrt(tf.reduce_mean(tf.square(label - data)))


def l1_error(label, data):
    with tf.name_scope('l1_error'):
        return tf.reduce_mean(tf.abs(label - data))

def poission_loss(label, data, *, compute_full_loss=False):
    with tf.name_scope('poission_loss'):
        label = tf.maximum(label, 0.0)
        data = tf.maximum(data, 0.0)
        # return log_possion_loss(tf.log(label), data)
        return tf.reduce_mean(tf.keras.losses.poisson(label, data))

def log_possion_loss(log_label, data, *, compute_full_loss=False):
    """
    log_label: log value of expectation (inference)
    data: Poission sample
    """
    with tf.name_scope('log_poission_loss'):
        data = tf.maximum(data, 0.0)
        return tf.reduce_mean(tf.nn.log_poisson_loss(data, log_label, compute_full_loss))

def get_loss_func(name):
    if name.lower() in ['mse', 'mean_square_error', 'l2']:
        return mean_square_error
    if name == 'poi':
        return poission_loss
    if name == 'l1':
        return l1_error
    raise ValueError("Unknown error name {}.".format(name))


class CombinedSupervisedLoss(Model):
    '''CombinedSupervisedLoss Block
    Args:
        inputs: Dict[str, Tensor/tf.Tensor] input.
            KEYS.TENSOR.INPUT: infer
            KEYS.TENSOR.LABEL: label
        loss_name: str
        loss_weights:
    '''
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            pass
        class CONFIG:
            LOSS_NAME = 'loss_name'
            LOSS_WEIGHTS = 'loss_weights'

    def __init__(self, name,
                 inputs,
                 loss_names=None,
                 loss_weights=None,
                 graph_info=None):
        super().__init__(
            name,
            inputs=inputs,
            graph_info=graph_info,
            config={
                self.KEYS.CONFIG.LOSS_NAME: loss_names,
                self.KEYS.CONFIG.LOSS_WEIGHTS: loss_names
            })

    def kernel(self, inputs):
        if not self.KEYS.TENSOR.LABEL in inputs:
            raise ValueError("{} is required for inputs of {}, but not found in inputs".format(
                self.KEYS.TENSOR.LABEL, __class__))
        if not self.KEYS.TENSOR.INPUT in inputs:
            raise ValueError("{} is required for inputs of {}, but not found in inputs".format(
                NodeKeys.INPUT, __class__))

        label = inputs[self.KEYS.TENSOR.LABEL]
        data = inputs[self.KEYS.TENSOR.INPUT]
        with tf.name_scope('combined_losses'):
            losses = []
            for n, w in zip(self.config(self.KEYS.CONFIG.LOSS_NAME),
                            self.config(self.KEYS.CONFIG.LOSS_WEIGHTS)):
                losses.append(get_loss_func(n)(label, data) * tf.constant(w))
            with tf.name_scope('summation'):
                loss = tf.add_n(losses)

        result = {self.KEYS.TENSOR.OUTPUT : loss}
        for i, n in enumerate(self.config(self.KEYS.CONFIG.LOSS_NAME)):
            result.update({'loss/' + n: losses[i]})

        return result
