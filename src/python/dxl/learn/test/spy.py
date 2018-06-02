from dxl.learn.core import Tensor, Variable, GraphInfo, NoOp
import uuid
import tensorflow as tf


class OpRunSpy:
    global_op_run_spy = None
    previous_assign = None

    def __init__(self, *, is_skip_global_order=False):
        with tf.variable_scope(self._scope()):
            if is_skip_global_order:
                n, o, g = self._init_skipped_global_order()
            else:
                n, o, g = self._init_with_global_order()

        self.nb_called, self.op, self.global_order = n, o, g

    def _init_with_global_order(self):
        nb_called = Variable('nb_called', initializer=0)
        global_order = Variable('global_order_snip', initializer=0)
        inc_global = self.get_or_set_nb_global_called().assign_add(1)
        inc_self = nb_called.assign_add(1, use_locking=True)
        with tf.control_dependencies(
            [inc_global.data, inc_self.data, self.previous_assign.data]):
            inc = NoOp()
        with tf.control_dependencies([inc.data]):
            assign_global_back = global_order.assign(
                self.get_or_set_nb_global_called())
            self.previous_assign = assign_global_back
        return nb_called, inc, assign_global_back

    def _init_skipped_global_order(self):
        nb_called = Variable(GraphInfo('nb_called', False), initializer=0)
        op = nb_called.assign_add(1, use_locking=True)
        global_order = None
        return nb_called, op, global_order

    def _scope(self):
        return 'runspy_scope/{}'.format('rs{}'.format(uuid.uuid4().hex))

    def reset(self):
        del self.global_op_run_spy
        del self.previous_assign

    def get_or_set_nb_global_called(self):
        if self.global_op_run_spy is None:
            self.global_op_run_spy = Variable(
                GraphInfo('nb_globall_called', 'runpy_scope', False),
                initializer=0)
            self.previous_assign = NoOp()
        return self.global_op_run_spy
