import click
import tensorflow as tf
from dxl.learn.core.distribute import Host, Cluster, Server, ThisHost, Master, make_distribute_host
from dxl.learn.core.session import SessionDistribute, ThisSession, make_distribute_session
from dxl.learn.core.tensor import TensorNumpyNDArray, Tensor, TensorVariable
import time


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
    with tf.variable_scope('scope_test'):
        t0 = TensorVariable([1], tf.float32, None,
                            master_host, {'name': 'zerovar'})
        t1 = TensorNumpyNDArray([1.0], None, master_host, {'name': 'testvar'})
        t1u = Tensor(t1.data + 1, t1.data_info, t1.host, {'name': 'testvarp1'})
        print(type(t1.data))
        print(t1.data)
        aop = tf.assign(t0.data, tf.constant([3.]))
        t2 = t0.copy_to(host2)
    make_distribute_session()
    if task == 0:
        # ThisSession.run(tf.global_variables_initializer())
        print(t1.run())
        print(t1.data)
        Server.join()
    if task == 1:
        print(t2.run())
        print(t2.data)
        print(t0.run())
        print(t0)
        print(ThisSession.run(aop))
        print('t0', t0.run())
        print('t2', t2.run())


@click.command()
@click.option('--job', '-j', help='Job')
@click.option('--task', '-t', help='task', type=int)
def cli(job, task):
    main(job, task)


if __name__ == "__main__":
    cli()
