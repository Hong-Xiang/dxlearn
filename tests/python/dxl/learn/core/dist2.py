import click
import tensorflow as tf
from dxl.learn.core import Host, ThisHost, Master, make_distribute_host, Server, ThisSession
from dxl.learn.core import make_distribute_session, DistributeGraphInfo, VariableInfo
from dxl.learn.core import TensorNumpyNDArray, Tensor, TensorVariable
from dxl.learn.model.on_collections import Summation

import time

from enum import Enum


class Tests(Enum):
    Basic = 0
    Summation = 1
    Sync = 2


def ptensor(t, name=None):
    print("DEBUG, name: {}, t.data: {}, t.run(): {}.".format(name, t.data, t.run()))


def main_sync(job, task):
    cfg = {"master": ["localhost:2221"],
           "worker": ["localhost:2222",
                      "localhost:2223"]}
    make_distribute_host(cfg, job, task, None, 'master', 0)
    master_host = Master.master_host()
    this_host = ThisHost.host()
    host0 = Host('worker', 0)
    host1 = Host('worker', 1)

    def sleep(ips):
        for i in range(5, 0, -1):
            time.sleep(1)
        return 0
    # hmi = DistributeGraphInfo(None, None, None, master_host)
    # tm = TensorNumpyNDArray([1.0], None,
    #                         DistributeGraphInfo.from_graph_info(hmi, name='t0'))
    # t0c = tm.copy_to(host0)
    # t1c = tm.copy_to(host1)
    # m_sum = Summation(name='summation', graph_info=DistributeGraphInfo(
    #     'summation', None, None, host0))([t0c, t1c])
    ops = tf.FIFOQueue(2, tf.bool, shapes=[],
                       name='barrier', shared_name='barrier')
    # ptensor(tm)
    if ThisHost.host() == master_host:
        join = ops.dequeue_many(2)
    else:
        signal = ops.enqueue(False)
    no = tf.constant('tmp')
    ops = [tf.Print(no, data=[no], message='Done_{}'.format(i), name='p_{}'.format(i))
           for i in range(3)]
    # ops.enqueue()
    make_distribute_session()
    if ThisHost.host() == master_host:
        ThisSession.run(join)
        print('Joined.')
        time.sleep(2)
        ThisSession.run(ops[0])
        # Server.join()
    elif ThisHost.host() == host0:
        ThisSession.run(signal)
        ThisSession.run(ops[1])
    elif ThisHost.host() == host1:
        time.sleep(3)
        ThisSession.run(signal)
        ThisSession.run(ops[2])


def main_summation(job, task):
    cfg = {"master": ["localhost:2221"],
           "worker": ["localhost:2222",
                      "localhost:2223"]}
    make_distribute_host(cfg, job, task, None, 'master', 0)
    master_host = Master.master_host()
    this_host = ThisHost.host()
    host0 = Host('worker', 0)
    host1 = Host('worker', 1)
    hmi = DistributeGraphInfo(None, None, None, master_host)
    tm = TensorNumpyNDArray([1.0], None,
                            DistributeGraphInfo.from_graph_info(hmi, name='t0'))
    t0c = tm.copy_to(host0)
    t1c = tm.copy_to(host1)
    m_sum = Summation(name='summation', graph_info=DistributeGraphInfo(
        'summation', None, None, host0))([t0c, t1c])
    make_distribute_session()
    if task == 0:
        ptensor(tm)
        Server.join()
    if task == 1:
        ptensor(tm, 'tm')
        ptensor(t0c, 't0c')
        ptensor(t1c, 't1c')
        ptensor(m_sum)


def main_basic(job, task):
    cfg = {"worker": ["localhost:2222",
                      "localhost:2223"]}
    make_distribute_host(cfg, job, task, None, 'worker', 0)
    master_host = Master.master_host()
    this_host = ThisHost.host()
    host1 = Host(job, 1)
    hmi = DistributeGraphInfo(None, None, None, master_host)
    with tf.variable_scope('scope_test'):
        t0 = TensorVariable(VariableInfo(None, [1], tf.float32),
                            hmi.update(name='v0'))
        aop = tf.assign(t0.data, tf.constant([3.]))
        t1 = TensorNumpyNDArray([1.0], None,
                                hmi.update(name='v1'))
        t1c = t1.copy_to(host1)
        t1p = Tensor(t1c.data + 1, t1c.data_info,
                     t1c.graph_info.update(name='t1_plus'))
    make_distribute_session()
    if task == 0:
        ptensor(t1)
        Server.join()
    if task == 1:
        ptensor(t1)
        ptensor(t1c)
        ptensor(t1p)
        ptensor(t0)
        ThisSession.run(aop)
        ptensor(t0)


@click.command()
@click.option('--job', '-j', help='Job')
@click.option('--task', '-t', help='task', type=int, default=0)
def cli(job, task):
    test = Tests.Sync
    if test == Tests.Basic:
        main_basic(job, task)
    if test == Tests.Summation:
        main_summation(job, task)
    if test == Tests.Sync:
        main_sync(job, task)


if __name__ == "__main__":
    cli()
