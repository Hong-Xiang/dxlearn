import numpy as np
import click
import tensorflow as tf
from dxl.learn.core import TensorNumpyNDArray, TensorVariable, Tensor, VariableInfo, DistributeGraphInfo, ThisSession, Session, Host, ThisHost, DistributeTask
from dxl.learn.core import make_distribute_host, make_distribute_session, Master, Barrier, Server
from typing import Iterable
import pdb
import time

from dxl.learn.model.tor_recon import ReconStep, ProjectionSplitter, EfficiencyMap
from dxl.learn.graph.tor_recon import GlobalGraph, LocalGraph


from .global_graph import GlobalGraph
from .local_graph import LocalGraph
from .utils import ImageInfo, print_tensor, DataInfo, load_data, print_info, constant_tensor
from .preprocess import preprocess as preprocess_tor


import json

MASTER_IP_SUFFIX = '118'
WORKERS_IO_SUFFIX = ['118', '110']
NB_PROCESS_PER_WORKER = 2
MODEL = 'TOR'

# Load configs

def load_dist_configs(config=None):
  if config is None:
    workers = []
    for worker in WORKERS_IO_SUFFIX:
      for p in range(2333, 2333 + NB_PROCESS_PER_WORKER):
        workers.append("192.168.1.{}:{}".format(worker, p))
    return {
        "master": ["192.168.1.{}:2221".format(MASTER_IP_SUFFIX)],
        "worker": workers
    }
  else:
    with open(config,'r') as fin:
      return json.load(fin)
  
def load_recon_configs(config=None):
  if config is not None:
    with open('config', 'r') as fin:
      c = json.load(fin)
  else:
    c = {
      'grid': [150, 150, 150],
      'center':[0., 0., 0.]
      'size':[150., 150., 150.]
      'map_file': './debug/map.npy'
      'x_lor_files': './debug/xlors.npy'
      'y_lor_files': './deubg/ylors.npy'
      'z_lor_files': './deubg/zlors.npy'
      'lor_ranges': None
      'lor_steps': None
    }
  image_info = ImageInfo(c['grid'], c['center'], c['size'])
  data_info = DataInfo(c['map_file'], {a: c['{}_lor_files'] for a in ['x', 'y', 'z']}, lor_ranges, lor_steps)
  return image_info, data_info

def task_init(job, task, config_file=None):
  task = DistributeTask(load_dist_configs(config_file))
  return task

def create_global_graph(task, image_info, x_value=1.0):
  x = np.ones(image_info.grid) * x_value
  gg = GlobalGraph(x, image_info, task.)
  print_info("Global graph created.")
  return gg

# Load datas
def load_local_data(data_info: DataInfo):
  print_info("Loading efficiency map from file: {}...".format(data_info.map_file()))
  emap = load_data(data_info.map_file())
  lors = {}
  for a in ['x', 'y', 'z']:
    msg = "Loading {} LORs from file: {}, with range: {}..."
    print_info(msg.format(data_info.map_file(), data_info.lor_range()))
    lors[a] = load_data(data_info.lor_file(a, data_info.lor_range()))
  print_info('Loading local data done.')
  return emap, lors

def create_local_graphs(global_graph: GlobalGraph, image_info, data_info: DataInfo, hosts):
  print_info("Creating local graph for worker {}...".format(ThisHost.host().task_index))
  emap, lors = load_local_data(data_info)
  with DistributeGraphInfo('datas', 'data_scope', None, ThisHost.host()).variable_scope():
    emap = constant_tensor(emap)
    lors = {a: constant_tensor(lors[a]) for a in lors}
  local_graphs = [LocalGraph(global_graph, h, emap, lors, image_info) for h in hosts]
  for lg in local_graphs:
    lg.make_recon_step()
  print_info("Local graph created.")
  return local_graph


def init_step(global_graph, hosts):
  master_host = Master.master_host()
  TK = global_graph.KEYS.TENSOR
  global_init_barrier = Barrier('global_init', [master_host], hosts, [[TK.INIT_OP]])
  master_op = global_init_barrier.barrier(master_host)
  workers_op = [global_init_barrier.barrier(h) for h in hosts]
  return master_op, workers_op


def recon_step(global_graph: GlobalGraph, local_graphs: Iterable[LocalGraph]):
  recon_local = [[g.tensor(g.KEYS.TENSOR.X_UPDATE)] for g in local_graphs]
  worker_hosts = [g.host for g in local_graphs]
  master_host = Master.master_host()
  calculate_barrier = Barrier(
      'calculate',
      worker_hosts, [master_host],
      task_lists=recon_local)
  TK = global_graph.KEYS.TENSOR
  master_barrier = calculate_barrier.barrier(master_host)
  worker_barriers = [calculate_barrier.barrier(h) for h in worker_hosts]
  return master_barrier, worker_barriers

def merge_step(global_graph, worker_hosts):
  """
  """
  merge_barrier = Barrier('merge', [Master.master_host()], worker_hosts,
                          [[self.merge_local_x()]])
  master_op = merge_barrier.barrier(Master.master_host())
  workers_ops = [merge_barrier.barrier(h) for h in worker_hosts]
  return master_op, workers_ops


def make_recon_local(global_graph, local_graphs):
  for g in local_graphs:
    g.copy_to_global(global_graph)
    g.recon_local()
  global_graph.x_update_by_merge()


def get_my_local_graph(local_graphs: Iterable[LocalGraph]):
  host = ThisHost.host()
  for g in local_graphs:
    if g.graph_info.host == host:
      return g
  raise KeyError("No local graph for {}.{} found".format(
      host.job, host.task_index))


def run_step(master_op, worker_ops, global_graph, local_graphs):
  if ThisHost.is_master():
    _op = master_op
  else:
    _op = get_my_local_graph(local_graphs)
  ThisSession.run(_op)

def run_init_ops(master_op, worker_ops, global_graph: GlobalGraph,
             local_graphs: Iterable[LocalGraph]):
  if ThisHost.is_master():
    ThisSession.run(master_op)
    print_tensor(global_graph.tensor(global_graph.KEYS.TENSOR.X), 'x:global')
  else:
    print_info('Pre intialization.')
    print_tensor(global_graph.tensor(global_graph.KEYS.TENSOR.X),
        'x:global direct fetch')
    tid = ThisHost.host().task_index
    ThisSession.run(worker_ops[tid])
    lg = get_my_local_graph(local_graphs)
    TK = lg.KEYS.TENSOR
    ptensor(lg.tensor(TK.X), 'x:local')
    # ptensor(lg.tensor(TK.SYSTEM_MATRIX), 'x:local')
  print_info('Intialization DONE. ==============================')


def run_recon_step(master_op, worker_ops, global_graph, local_graphs):
  if ThisHost.is_master():
    print_info('PRE RECON')
    print_tensor(global_graph.tensor('x'))
    print_info('START RECON')
    ThisSession.run(master_op)
    print_info('POST RECON')
    ptensor(global_graph.tensor('x'), 'x:global')
  else:
    print_info('PRE RECON')
    lg = get_my_local_graph(local_graphs)
    TK = lg.KEYS.TENSOR
    print_tensor(lg.tensor(TK.X), 'x:local')
    print_info('POST RECON')
    ThisSession.run(worker_ops[ThisHost.host().task_index])
    # ptensor(lg.tensor(TK.X_UPDATE), 'x:update')
    print_tensor(lg.tensor(TK.X_RESULT), 'x:result')
    print_tensor(lg.tensor(TK.X_GLOBAL_BUFFER), 'x:global_buffer')
    # ThisSession.run(ThisHost.host().task_index)

def run_merge_step(master_op, worker_ops, global_graph, local_graphs):
  if ThisHost.is_master():
    ThisSession.run(master_op)


def full_step_run(m_op,
                  w_ops,
                  global_graph,
                  local_graphs,
                  nb_iter=0,
                  verbose=0):
  if verbose > 0:
    print('PRE RECON {}'.format(nb_iter))
    lg = None
    if ThisHost.is_master():
      TK = global_graph.KEYS.TENSOR
      ptensor(global_graph.tensor(TK.X), 'x:global')
    else:
      lg = get_my_local_graph(local_graphs)
      TK = lg.KEYS.TENSOR
      ptensor(lg.tensor(TK.X), 'x:local')
  print('START RECON {}'.format(nb_iter))
  if ThisHost.is_master():
    ThisSession.run(m_op)
  else:
    ThisSession.run(w_ops[ThisHost.host().task_index])
  if verbose > 0:
    print('POST RECON {}'.format(nb_iter))
    if ThisHost.is_master():
      TK = global_graph.KEYS.TENSOR
      ptensor(global_graph.tensor(TK.X), 'x:global')
    else:
      lg = get_my_local_graph(local_graphs)
      TK = lg.KEYS.TENSOR
      ptensor(lg.tensor(TK.X), 'x:local')


def main(job, task):
  hosts, hmi = dist_init(job, task)
  global_graph = init_global(hmi)
  local_graphs = init_local(global_graph, hosts)

  m_op_init, w_ops_init = global_graph.init_op(local_graphs)
  make_recon_local(global_graph, local_graphs)
  m_op_rec, w_ops_rec = global_graph.recon_step(local_graphs, hosts)
  m_op, w_ops = global_graph.merge_step(m_op_rec, w_ops_rec, hosts)
  # global_tensors = {'x': x_t, 'y': y_t, 'sm': sm_t, 'em': e_t}
  # g2l_init, update_global, x_g2l, x_l, y_l, sm_l, em_l = recon_init(
  #     x_t, y_t, sm_t, e_t, hosts, x_init)
  # gop, l_ops = recon_step(update_global, x_g2l, x_l, y_l, sm_l, em_l, hosts)
  # init_op = make_init_op(g2l_init, hosts)

  make_distribute_session()

  tf.summary.FileWriter('./graph', ThisSession.session().graph)
  print('|DEBUG| Make Graph done.')

  init_run(m_op_init, w_ops_init, global_graph, local_graphs)
  ptensor(global_graph.tensor(global_graph.KEYS.TENSOR.X))

  # time.sleep(5)
  # recon_run(m_op_rec, w_ops_rec, global_graph, local_graphs)
  start_time = time.time()
  for i in range(5):
    full_step_run(m_op, w_ops, global_graph, local_graphs, i)
    end_time = time.time()
    delta_time = end_time - start_time
    msg = "the step running time is:{}".format(delta_time / (i + 1))
    print(msg)
    if ThisHost.is_master():
      res = global_graph.tensor(global_graph.KEYS.TENSOR.X).run()
      np.save('./gpu_all/recon_{}.npy'.format(i), res)
  ptensor(global_graph.tensor(global_graph.KEYS.TENSOR.X))
  # full_step_run(m_op, w_ops, global_graph, local_graphs, 1)
  # full_step_run(m_op, w_ops, global_graph, local_graphs, 2)
  # if ThisHost.is_master():
  #     ThisSession.run(gop)
  # else:
  #     ThisSession.run(l_ops[ThisHost.host().task_index])
  if ThisHost.is_master():
    # res = global_graph.tensor(global_graph.KEYS.TENSOR.X).run()
    # np.save('recon.npy', res)
    pass
  # print('|DEBUG| JOIN!')
  # Server.join()
  print('DONE!')
  end_time = time.time()
  delta_time = end_time - start_time
  msg = "the total running time is:{}".format(delta_time)
  print(msg)
  if ThisHost.is_master():
    with open('time_cost.txt', 'w') as fout:
      print(msg, file=fout)
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


@click.command()
@click.option('--job', '-j', help='Job')
@click.option('--task', '-t', help='task', type=int, default=0)
def cli(job, task):
  main(job, task)


if __name__ == "__main__":
  cli()
