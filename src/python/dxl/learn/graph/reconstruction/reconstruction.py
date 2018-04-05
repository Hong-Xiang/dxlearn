"""
Reconstruction with memory optimized:

sample code :: python

  import click
  import logging
  from dxl.learn.graph.reconstruction.reconstruction import main

  logger = logging.getLogger('dxl.learn.graph.reconstruction')
  logger.setLevel(logging.DEBUG)



  @click.command()
  @click.option('--job', '-j', help='Job')
  @click.option('--task', '-t', help='task', type=int, default=0)
  def cli(job, task):
    main(job, task)


  if __name__ == "__main__":
    cli()

"""
import numpy as np
import click
import tensorflow as tf
from dxl.learn.core import make_distribute_session, Master, Barrier, ThisHost, ThisSession, Tensor
from typing import Iterable
import pdb
import time

from .master import MasterGraph
from .worker import WorkerGraphLOR
from .utils import ImageInfo, print_tensor, DataInfo, load_data, logger, constant_tensor, load_reconstruction_configs, load_local_data
from .utils import debug_tensor
from .preprocess import preprocess as preprocess_tor
from .distribute import DistributeTask, load_cluster_configs
import json

from tqdm import tqdm


def task_init(job, task, config=None):
  t = DistributeTask(load_cluster_configs(config))
  t.cluster_init(job, task)
  return t


def create_master_graph(task: DistributeTask, x):
  mg = MasterGraph(x, task.nb_workers(), task.ginfo_master())
  task.add_master_graph(mg)
  logger.info("Global graph created.")
  return mg


def create_worker_graphs(task: DistributeTask, image_info,
                         data_info: DataInfo):
  for i in range(task.nb_workers()):
    logger.info("Creating local graph for worker {}...".format(i))
    task.add_worker_graph(
        WorkerGraphLOR(
            task.master_graph,
            image_info,
            {a: data_info.lor_shape(a, i)
             for a in ['x', 'y', 'z']},
            i,
            task.ginfo_worker(i),
        ))
  logger.info("All local graph created.")
  return task.worker_graphs


def bind_local_data(data_info, task: DistributeTask, task_index=None):
  if task_index is None:
    task_index = ThisHost.host().task_index
  if ThisHost.is_master():
    logger.info("On Master node, skip bind local data.")
    return
  else:
    logger.info("On Worker node, local data for worker {}.".format(task_index))
    emap, lors = load_local_data(data_info, task_index)
    task.worker_graphs[task_index].assign_efficiency_map_and_lors(emap, lors)


def make_init_step(task: DistributeTask, name='init'):
  init_barrier = Barrier(name, task.hosts, [task.master_host],
                         [[g.tensor(g.KEYS.TENSOR.INIT)]
                          for g in task.worker_graphs])
  master_op = init_barrier.barrier(task.master_host)
  worker_ops = [init_barrier.barrier(h) for h in task.hosts]
  task.add_step(name, master_op, worker_ops)
  return name


def make_recon_step(task: DistributeTask, name='recon'):
  recons = [[g.tensor(g.KEYS.TENSOR.UPDATE)] for g in task.worker_graphs]
  calculate_barrier = Barrier(
      name, task.hosts, [task.master_host], task_lists=recons)
  master_op = calculate_barrier.barrier(task.master_host)
  worker_ops = [calculate_barrier.barrier(h) for h in task.hosts]
  task.add_step(name, master_op, worker_ops)
  return name


def make_merge_step(task: DistributeTask, name='merge'):
  """
  """
  merge_op = task.master_graph.tensor(task.master_graph.KEYS.TENSOR.UPDATE)
  merge_barrier = Barrier(name, [task.master_host], task.hosts, [[merge_op]])
  master_op = merge_barrier.barrier(task.master_host)
  worker_ops = [merge_barrier.barrier(h) for h in task.hosts]
  task.add_step(name, master_op, worker_ops)
  return name


def run_and_save_if_is_master(x, path):
  if ThisHost.is_master():
    if isinstance(x, Tensor):
      x = x.data
    result = ThisSession.run(x)
    np.save(path, result)


# def make_recon_local(global_graph, local_graphs):
#   for g in local_graphs:
#     g.copy_to_global(global_graph)
#     g.recon_local()
#   global_graph.x_update_by_merge()

# def get_my_local_graph(local_graphs: Iterable[LocalGraph]):
#   host = ThisHost.host()
#   for g in local_graphs:
#     if g.graph_info.host == host:
#       return g
#   raise KeyError("No local graph for {}.{} found".format(
#       host.job, host.task_index))

# def run_step(master_op, worker_ops, global_graph, local_graphs):
#   if ThisHost.is_master():
#     _op = master_op
#   else:
#     _op = get_my_local_graph(local_graphs)
#   ThisSession.run(_op)

# def run_init_ops(master_op, worker_ops, global_graph: GlobalGraph,
#                  local_graphs: Iterable[LocalGraph]):
#   if ThisHost.is_master():
#     ThisSession.run(master_op)
#     print_tensor(global_graph.tensor(global_graph.KEYS.TENSOR.X), 'x:global')
#   else:
#     logger.info('Pre intialization.')
#     print_tensor(
#         global_graph.tensor(global_graph.KEYS.TENSOR.X),
#         'x:global direct fetch')
#     tid = ThisHost.host().task_index
#     ThisSession.run(worker_ops[tid])
#     lg = get_my_local_graph(local_graphs)
#     TK = lg.KEYS.TENSOR
#     ptensor(lg.tensor(TK.X), 'x:local')
#     # ptensor(lg.tensor(TK.SYSTEM_MATRIX), 'x:local')
#   logger.info('Intialization DONE. ==============================')

# def run_recon_step(master_op, worker_ops, global_graph, local_graphs):
#   if ThisHost.is_master():
#     logger.info('PRE RECON')
#     print_tensor(global_graph.tensor('x'))
#     logger.info('START RECON')
#     ThisSession.run(master_op)
#     logger.info('POST RECON')
#     ptensor(global_graph.tensor('x'), 'x:global')
#   else:
#     logger.info('PRE RECON')
#     lg = get_my_local_graph(local_graphs)
#     TK = lg.KEYS.TENSOR
#     print_tensor(lg.tensor(TK.X), 'x:local')
#     logger.info('POST RECON')
#     ThisSession.run(worker_ops[ThisHost.host().task_index])
#     # ptensor(lg.tensor(TK.X_UPDATE), 'x:update')
#     print_tensor(lg.tensor(TK.X_RESULT), 'x:result')
#     print_tensor(lg.tensor(TK.X_GLOBAL_BUFFER), 'x:global_buffer')
#     # ThisSession.run(ThisHost.host().task_index)

# def run_merge_step(master_op, worker_ops, global_graph, local_graphs):
#   if ThisHost.is_master():
#     ThisSession.run(master_op)

# def full_step_run(m_op,
#                   w_ops,
#                   global_graph,
#                   local_graphs,
#                   nb_iter=0,
#                   verbose=0):
#   if verbose > 0:
#     print('PRE RECON {}'.format(nb_iter))
#     lg = None
#     if ThisHost.is_master():
#       TK = global_graph.KEYS.TENSOR
#       ptensor(global_graph.tensor(TK.X), 'x:global')
#     else:
#       lg = get_my_local_graph(local_graphs)
#       TK = lg.KEYS.TENSOR
#       ptensor(lg.tensor(TK.X), 'x:local')
#   print('START RECON {}'.format(nb_iter))
#   if ThisHost.is_master():
#     ThisSession.run(m_op)
#   else:
#     ThisSession.run(w_ops[ThisHost.host().task_index])
#   if verbose > 0:
#     print('POST RECON {}'.format(nb_iter))
#     if ThisHost.is_master():
#       TK = global_graph.KEYS.TENSOR
#       ptensor(global_graph.tensor(TK.X), 'x:global')
#     else:
#       lg = get_my_local_graph(local_graphs)
#       TK = lg.KEYS.TENSOR
#       ptensor(lg.tensor(TK.X), 'x:local')

# def main(job, task):
#   hosts, hmi = dist_init(job, task)
#   global_graph = init_global(hmi)
#   local_graphs = init_local(global_graph, hosts)

#   m_op_init, w_ops_init = global_graph.init_op(local_graphs)
#   make_recon_local(global_graph, local_graphs)
#   m_op_rec, w_ops_rec = global_graph.recon_step(local_graphs, hosts)
#   m_op, w_ops = global_graph.merge_step(m_op_rec, w_ops_rec, hosts)
#   # global_tensors = {'x': x_t, 'y': y_t, 'sm': sm_t, 'em': e_t}
#   # g2l_init, update_global, x_g2l, x_l, y_l, sm_l, em_l = recon_init(
#   #     x_t, y_t, sm_t, e_t, hosts, x_init)
#   # gop, l_ops = recon_step(update_global, x_g2l, x_l, y_l, sm_l, em_l, hosts)
#   # init_op = make_init_op(g2l_init, hosts)

#   make_distribute_session()

#   tf.summary.FileWriter('./graph', ThisSession.session().graph)
#   print('|DEBUG| Make Graph done.')

#   init_run(m_op_init, w_ops_init, global_graph, local_graphs)
#   ptensor(global_graph.tensor(global_graph.KEYS.TENSOR.X))

#   # time.sleep(5)
#   # recon_run(m_op_rec, w_ops_rec, global_graph, local_graphs)
#   start_time = time.time()
#   for i in range(5):
#     full_step_run(m_op, w_ops, global_graph, local_graphs, i)
#     end_time = time.time()
#     delta_time = end_time - start_time
#     msg = "the step running time is:{}".format(delta_time / (i + 1))
#     print(msg)
#     if ThisHost.is_master():
#       res = global_graph.tensor(global_graph.KEYS.TENSOR.X).run()
#       np.save('./gpu_all/recon_{}.npy'.format(i), res)
#   ptensor(global_graph.tensor(global_graph.KEYS.TENSOR.X))
#   # full_step_run(m_op, w_ops, global_graph, local_graphs, 1)
#   # full_step_run(m_op, w_ops, global_graph, local_graphs, 2)
#   # if ThisHost.is_master():
#   #     ThisSession.run(gop)
#   # else:
#   #     ThisSession.run(l_ops[ThisHost.host().task_index])
#   if ThisHost.is_master():
#     # res = global_graph.tensor(global_graph.KEYS.TENSOR.X).run()
#     # np.save('recon.npy', res)
#     pass
#   # print('|DEBUG| JOIN!')
#   # Server.join()
#   print('DONE!')
#   end_time = time.time()
#   delta_time = end_time - start_time
#   msg = "the total running time is:{}".format(delta_time)
#   print(msg)
#   if ThisHost.is_master():
#     with open('time_cost.txt', 'w') as fout:
#       print(msg, file=fout)
# import imageio

# img = np.load('recon.npy')
# img = img.reshape([150, 150, 150])
# imgslice = img[75,:,:]
# imageio.imwrite('recon.png', imgslice)

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


def main(job, task_index):
  logger.info("Start reconstruction job: {}, task_index: {}.".format(
      job, task_index))
  task = task_init(job, task_index)
  image_info, data_info = load_reconstruction_configs('./recon.json')
  logger.info("Local data_info:\n" + str(data_info))
  create_master_graph(task, np.ones(image_info.grid, dtype=np.float32))
  create_worker_graphs(task, image_info, data_info)
  bind_local_data(data_info, task)
  init_step = make_init_step(task)
  recon_step = make_recon_step(task)
  merge_step = make_merge_step(task)
  make_distribute_session()
  task.run_step_of_this_host(init_step)
  logger.info('STEP: {} done.'.format(init_step))
  nb_steps = 10
  for i in tqdm(range(nb_steps), ascii=True):
    task.run_step_of_this_host(recon_step)
    logger.info('STEP: {} done.'.format(recon_step))
    task.run_step_of_this_host(merge_step)
    logger.info('STEP: {} done.'.format(merge_step))
    run_and_save_if_is_master(
        task.master_graph.tensor('x'),
        './debug/mem_lim_result_{}.npy'.format(i))
  logger.info('Recon {} steps done.'.format(nb_steps))
  time.sleep(5)


@click.command()
@click.option('--job', '-j', help='Job')
@click.option('--task', '-t', help='task', type=int, default=0)
def cli(job, task):
  main(job, task)


if __name__ == "__main__":
  cli()
