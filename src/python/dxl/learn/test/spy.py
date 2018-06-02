from dxl.learn.core import Tensor, Variable, GraphInfo
import uuid


def _name_and_scope():
    name = 'runspy{}'.format(uuid.uuid4().hex)
    scope = 'runspy_scope'
    return name, scope


def tensor_run_spy():
    name, scope = _name_and_scope()
    value = Variable(GraphInfo(name, scope, False), initializer=0)
    added = value.assign_add(1)
    return added, value
