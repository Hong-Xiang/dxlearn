from doufo import all_isinstance
import tensorflow as tf

__all__ = ['merge_ops']

def merge_ops(ops):
    if all_isinstance(ops, (tf.Tensor, tf.Operation, tf.Variable)):
        with tf.control_dependencies(ops):
            return tf.no_op()
    raise TypeError(f"No implementation for merge_ops({ops}) yet.")
