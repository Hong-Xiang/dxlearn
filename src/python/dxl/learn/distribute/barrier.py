import tensorflow as tf
from .tensor import Tensor, NoOp




def barrier_single(name, nb_signal, nb_join, task=None, id_join=None):
    """
    `name`: global unique name of barrier.
    `task`: for signal hosts only
    `id_join`: for join hosts only

    Returns:
        A NoOp object as an barrier op.
    """
    name = str(name)
    with tf.name_scope(name):
        with tf.name_scope('queues'):
            names = ["{}_{}".format(name, i) for i in range(nb_join)]
            queues = [
                tf.FIFOQueue(nb_signal, tf.bool, [], name=n, shared_name=n)
                for n in names
            ]

        with tf.name_scope('join'):
            if id_join is not None:
                join_op = queues[id_join].dequeue_many(nb_signal)
            else:
                join_op = tf.no_op()

        with tf.name_scope('signal'):
            if task is not None:
                if isinstance(task, Tensor):
                    task = task.data
                with tf.control_dependencies([task]):
                    _ops = [q.enqueue(False) for q in queues]
                with tf.control_dependencies(_ops):
                    signal_op = tf.no_op()
            else:
                signal_op = tf.no_op()

        with tf.name_scope('merged_op'):
            with tf.control_dependencies([join_op, signal_op]):
                return NoOp()
