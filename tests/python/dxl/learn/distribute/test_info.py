from dxl.learn.test import TestCase
from dxl.learn.distribute import Host, DistributeGraphInfo
import tensorflow as tf


class TestDistributeGraphInfo(TestCase):
    def test_host(self):
        h = Host('master', 0)
        dgi = DistributeGraphInfo('g', h)
        assert dgi.host == h

    def test_scope(self):
        job = 'master'
        task_index = 1
        h = Host(job, task_index)
        dgi = DistributeGraphInfo('g', h)
        with dgi.variable_scope():
            x = tf.get_variable('x', [])
            assert x.device == '/job:{}/task:{}'.format(job, task_index)
