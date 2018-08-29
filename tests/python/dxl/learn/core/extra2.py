import click
import tensorflow as tf
# from dxl.learn.core.distribute import Host, Cluster, Server, ThisHost, Master, make_distribute_host
from dxl.learn.core import Host, ThisHost, Master, make_distribute_host, Server, ThisSession
# from dxl.learn.core.depsession import SessionDistribute, ThisSession, make_distribute_session
from dxl.learn.core import make_distribute_session, DistributeGraphInfo, VariableInfo
from dxl.learn.core import TensorNumpyNDArray, Tensor, TensorVariable

import time


def ptensor(t):
    print("t.data: {}, t.run(): {}.".format(t.data, t.run()))


def main(job, task):
    tf.logging.set_verbosity(0)
    cfg = {"worker": ["localhost:2222",
                      "localhost:2223"]}
    make_distribute_host(cfg, job, task, None, 'worker', 0)
    # # if task == 1:
    #     # time.sleep(10)
    # with tf.device(Master.master_host().device_prefix()):
    #     with tf.variable_scope('test'):
    #         t1 = tf.get_variable('var', [], tf.float32)
    master_host = Master.master_host()
    this_host = ThisHost.host()
    host2 = Host(job, 1)
    hmi = DistributeGraphInfo(None, None, None, master_host)
    with tf.variable_scope('scope_test'):
        t0 = TensorVariable(VariableInfo(None, [1], tf.float32),
                            DistributeGraphInfo.from_(hmi, name='t1'))
        aop = tf.assign(t0.data, tf.constant([3.]))
        t1 = TensorNumpyNDArray([1.0], None,
                                DistributeGraphInfo.from_(hmi, name='t1_copy'))
        t1c = t1.copy_to(host2)
        t1p = Tensor(t1c.data + 1, t1c.data_info, DistributeGraphInfo.from_(t1c.graph_info, name='t1_plus'))
        # t2 = t0.copy_to(host2)
    make_distribute_session()
    if task == 0:
        # ThisSession.run(tf.global_variables_initializer())
        ptensor(t1)
        Server.join()
    if task == 1:
        ptensor(t1)
        ptensor(t1c)
        ptensor(t1p)
        # print(t2.run())
        # print(t2.data)
        # print(t0.run())
        # print(t0)
        ptensor(t0)
        print(ThisSession.run(aop))
        ptensor(t0)
        # print('t2', t2.run())


@click.command()
@click.option('--job', '-j', help='Job')
@click.option('--task', '-t', help='task', type=int)
def cli(job, task):
    main(job, task)


if __name__ == "__main__":
    cli()
