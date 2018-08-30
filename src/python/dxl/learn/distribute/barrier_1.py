import tensorflow as tf
from .tensor import Tensor, NoOp
from ..graph import Graph


class Barrier(Graph):
    def __init__(self, info, nb_signal, nb_join, task=None, id_join=None):
        """
        `name`: global unique name of barrier.
        `task`: for signal hosts only
        `id_join`: for join hosts only

        Returns:
            A NoOp object as an barrier op.
        """
        super().__init__(info, tensors={'task': task},
                         config={'nb_signal': nb_signal, 'nb_join': nb_join, 'id_join': id_join})

    def make_queues(self):
        with tf.name_scope('queues'):
            names = ["{}_{}".format(str(self.name), i) for i in range(self.config('nb_join'))]
            queues = [
                tf.FIFOQueue(self.config('nb_join'), tf.bool, [], name=n, shared_name=n)
                for n in names
            ]
        return queues

    def make_join(self, queues):
        with tf.name_scope('join'):
            if self.config('id_join') is not None:
                join_op = queues[self.config('id_join')].dequeue_many(self.config('nb_signal'))
            else:
                join_op = tf.no_op()
        return join_op

    def make_signal(self, queues):
        with tf.name_scope('signal'):
            task = self.tensor('task')
            if task is not None:
                if isinstance(task, Tensor):
                    task = task.data
                with tf.control_dependencies([task]):
                    _ops = [q.enqueue(False) for q in queues]
                with tf.control_dependencies(_ops):
                    signal_op = tf.no_op()
            else:
                signal_op = tf.no_op()
        return signal_op

    def kernel(self):
        queues = self.make_queues()
        join = self.make_join(queues)
        signal = self.make_signal(queues)
        with tf.name_scope('merged_op'):
            with tf.control_dependencies([join_op, signal_op]):
                self.tensors[self.KEYS.TENSOR.MAIN] = NoOp()
