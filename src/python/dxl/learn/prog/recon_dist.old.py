import numpy as np
import click
import tensorflow as tf
from dxl.learn.core import TensorNumpyNDArray, TensorVariable, Tensor, VariableInfo, DistributeGraphInfo, ThisSession, Session, Host, ThisHost
from dxl.learn.core import make_distribute_host, make_distribute_session, Master, Barrier, Server
from dxl.learn.model.recon import ReconStep, ProjectionSplitter, EfficiencyMap
from dxl.learn.model.on_collections import Summation
import pdb
import time

NB_WORKERS = 2


def ptensor(t, name=None):
    print("|DEBUG| name: {} | data: {} | run() {} |.".format(name, t.data, t.run()))


def dist_init(job, task):
    cfg = {"master": ["localhost:2221"],
           "worker": ["localhost:2222",
                      "localhost:2223"]}
    make_distribute_host(cfg, job, task, None, 'master', 0)
    master_host = Master.master_host()
    hosts = [Host('worker', i) for i in range(NB_WORKERS)]
    hmi = DistributeGraphInfo(None, None, None, master_host)
    return hosts, hmi




def init(hmi):
    root = '/mnt/gluster/hongxwing/recon_test/'
    phantom = np.load(root + 'phantom_64.0.npy')
    x = phantom.reshape([phantom.size, 1]).astype(np.float32)
    system_matrix = np.load(root + 'system_matrix_64.npy').astype(np.float32)
    y = np.matmul(system_matrix, x).astype(np.float32)
    effmap = np.matmul(system_matrix.T, np.ones(y.shape)).astype(np.float32)
    x_ = np.ones(x.shape, np.float32)

    x_var_info = VariableInfo(None, x_.shape, tf.float32)
    x_t = TensorVariable(x_var_info, hmi.update(name='x_t'))
    x_init = x_t.assign(TensorNumpyNDArray(
        x_, None, x_t.graph_info.update(name='x_init')))
    e_t = TensorNumpyNDArray(
        effmap, None, x_t.graph_info.update(name='effmap'))
    y_t = TensorNumpyNDArray(y, None, x_t.graph_info.update(name='y'))
    sm_t = TensorNumpyNDArray(
        system_matrix, None, x_t.graph_info.update(name='system_matrix'))
    return x_t, y_t, sm_t, e_t, x_init


def split(y_t, sm_t):
    spt = ProjectionSplitter('splitter', NB_WORKERS,
                             y_t.graph_info.update(name=None))
    y_ts = spt(y_t)
    sm_ts = spt(sm_t)
    y_ts = [y_ts['slice_{}'.format(i)] for i in range(NB_WORKERS)]
    sm_ts = [sm_ts['slice_{}'.format(i)] for i in range(NB_WORKERS)]
    return y_ts, sm_ts


def copy_to_local(x_t: TensorVariable, y_t: TensorNumpyNDArray, sm_t: Tensor, e_t: Tensor, host):
    x_t_local_copy, x_t_local = x_t.copy_to(host, True)
    y_t_local = y_t.copy_to(host)
    sm_t_local = sm_t.copy_to(host)
    e_t_local = e_t.copy_to(host)
    return x_t_local_copy, x_t_local, y_t_local, sm_t_local, e_t_local


def copy_to_global(x_t_local):
    x_t_global_copy, x_t_global_new = x_t_local.copy_to(
        Master.master_host(), True)
    return x_t_global_copy, x_t_global_new


def recon_local(x_t, x_t_var: TensorVariable, y_t, sm_t, e_t, host: Host):
    x_n = ReconStep('recon_step_{}'.format(host.task_index),
                    x_t, y_t, sm_t, e_t,
                    x_t.graph_info.update(name=None))()
    x_t_var.assign(x_n)
    return x_t_var


def copy_back(x_t_global: TensorVariable, x_t_local2globals):
    with tf.control_dependencies([x_t_local2globals]):
        sm = Summation('summation', x_t.graph_info.update(name=None))
        x_m = sm(x_t_globals)
        x_t_update = x_t_global.assign(x_m)
        return x_t_update


def recon_init(x: TensorVariable, y: TensorNumpyNDArray, sm: TensorNumpyNDArray, em: TensorNumpyNDArray, worker_hosts, x_init):
    x_global2local = []
    x_local = []
    y_local = []
    sm_local = []
    em_local = []
    x_local2global = []
    x_global_buffer = []
    for h in worker_hosts:
        x_global2local_, x_local_, y_local_, sm_local_, em_local_ = copy_to_local(
            x, y, sm, em, h)
        x_global2local.append(x_global2local_)
        x_local.append(x_local_)
        y_local.append(y_local_)
        sm_local.append(sm_local_)
        em_local.append(em_local_)
        x_local2global_, x_global_buffer_ = copy_to_global(x_local_)
        x_local2global.append(x_local2global_)
        x_global_buffer.append(x_global_buffer_)
    deps = x_global2local + y_local + sm_local + em_local
    deps = [t.data for t in deps]
    with tf.control_dependencies([x_init.data]):
        with tf.control_dependencies(deps):
            g2l_init = tf.no_op()
    with tf.control_dependencies([t.data for t in x_global2local]):
        update_local = tf.no_op()
    with tf.control_dependencies([t.data for t in x_local2global]):
        update_global = tf.no_op()
    return g2l_init, update_global, x_global2local, x_local, y_local, sm_local, em_local


def recon_step(update_global, x_global2local, x_local, y_local, sm_local, em_local, worker_hosts):
    x_local_steps = []
    l_ops = []
    master_host = Master.master_host()
    for x, xv, y, sm, em, h in zip(x_global2local, x_local, y_local, sm_local, em_local, worker_hosts):
        x_local_steps.append(recon_local(x, xv, y, sm, em, h))
    calculate_barrier = Barrier('calculate', worker_hosts, [master_host],
                                task_lists=[x_local_steps])
    with tf.control_dependencies([calculate_barrier.barrier(master_host)]):
        with tf.control_dependencies([update_global]):
            gop = tf.no_op()
    merge_barrier = Barrier('merge', [Master.master_host()],
                            worker_hosts, [[gop]])
    l_ops = []
    for h in worker_hosts:
        with tf.control_dependencies([merge_barrier.barrier(h)]):
            l_ops.append(tf.no_op())
    return gop, l_ops


def make_init_op(global2local_init, hosts):
    master_host = Master.master_host()
    init_barrier = Barrier('init', [master_host], hosts, [global2local_init])
    if ThisHost.is_master():
        op_ = init_barrier.barrier(master_host)
    else:
        op_ = init_barrier.barrier(ThisHost.host())
    return op_


def init_run(init_op, global_tensors):
    if ThisHost.is_master():
        ThisSession.run(init_op)
        ptensor(global_tensors['x'], 'x:global')
    else:
        print('PRE INIT Barrier')
        ptensor(global_tensors['x'], 'x:global direct fetch')
        ThisSession.run(init_op)
        ptensor(x_l[ThisHost.host().task_index], 'x:local')
    print('INIT DONE.')


def recon_run(gop, l_ops):
    if ThisHost.is_master():
        ThisSession.run(gop)
    else:
        ThisSession.run(ThisHost.host().task_index)


def main(job, task):
    hosts, hmi = dist_init(job, task)
    x_t, y_t, sm_t, e_t, x_init = init(hmi)
    global_tensors = {'x': x_t, 'y': y_t, 'sm': sm_t, 'em': e_t}
    g2l_init, update_global, x_g2l, x_l, y_l, sm_l, em_l = recon_init(
        x_t, y_t, sm_t, e_t, hosts, x_init)
    gop, l_ops = recon_step(update_global, x_g2l, x_l, y_l, sm_l, em_l, hosts)
    init_op = make_init_op(g2l_init, hosts)
    make_distribute_session()
    print('|DEBUG| Make Graph done.')
    init_run(init_op, global_tensors)
    if ThisHost.is_master():
        ThisSession.run(gop)
    else:
        ThisSession.run(l_ops[ThisHost.host().task_index])
    print('|DEBUG| JOIN!')
    Server.join()

    # y_ts, sm_ts = split(y_t, sm_t)
    # make_distribute_session
    # x_init.run()
    # res = ThisSession.run(x_t.data)
    # x_ns = []
    # for i in range(NB_WORKERS):

    #     x_ns.append(recon(x_t, y_ts[i], sm_ts[i],  e_t, i))

    # x_n = sm(x_ns)
    # x_update = x_t.assign(x_n)
    # for i in range(100):
    #     x_update.run()
    # res = x_t.run()
    # print(res)
    # np.save('recon.npy', res)


@click.command()
@click.option('--job', '-j', help='Job')
@click.option('--task', '-t', help='task', type=int, default=0)
def cli(job, task):
    main(job, task)


if __name__ == "__main__":
    cli()
