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
  @click.option('--config', '-c', help='config file')
  def cli(job, task, config):
    main(job, task, config)


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
import jsonc

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


def main(job, task_index, config=None):
  if config is None:
    config = './recon.json'
  logger.info("Start reconstruction job: {}, task_index: {}.".format(
      job, task_index))
  task = task_init(job, task_index)
  image_info, data_info = load_reconstruction_configs(config)
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
