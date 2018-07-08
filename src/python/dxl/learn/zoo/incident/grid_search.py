import shutil
import os
import json
from pathlib import Path
from uuid import uuid4
import click


def generate_configs():
    batch_size = [32, 128, 512]
    learning_rate = [1e-3, 1e-4, 1e-5]
    nb_units_single_layer = [64, 128, 256]
    nb_layers = [5, 10, 20]
    nb_units = [[ns]*nl for ns in nb_units_single_layer for nl in nb_layers]
    configs = []
    for b in batch_size:
        for l in learning_rate:
            for nu in nb_units:
                config = {
                    'batch_size': b,
                    'learning_rate': l*b/32,
                    'nb_units': nu
                }
                config['nb_hits'] = 2
                config['path_table'] = '../../data/gamma_photon_5.h5'
                configs.append(config)
    return configs


def get_subdir(config):
    return 'b{}_l{}_n{}_lr{}'.format(config['batch_size'],
                                     len(config['nb_units']),
                                     config['nb_units'][0],
                                     config['learning_rate'])


def make_subdir(config):
    os.mkdir(get_subdir(config))
    with open(Path(get_subdir(config))/'config.json', 'w') as fout:
        json.dump(config, fout, separators=(',', ': '), indent=4)


def make_run_files(nb_gpu, nb_task_per_gpu):
    nb_task = nb_gpu * nb_task_per_gpu
    configs = generate_configs()
    for c in configs:
        make_subdir(c)
    dirs = [[] for _ in range(nb_task)]
    for i, c in enumerate(configs):
        dirs[i % nb_task].append(get_subdir(c))
    valid_gpus = [0, 2]
    for i in range(nb_task):
        with open('run_{}.sh'.format(i), 'w') as fout:
            print('export CUDA_VISIBLE_DEVICES="{}"'.format(
                valid_gpus[i % nb_gpu]), file=fout)
            for d in dirs[i]:
                print('cd {}'.format(d), file=fout)
                print('learn zoo incident train -c config.json', file=fout)
                print('cd ..', file=fout)


@click.command()
@click.option('--nb-gpu', type=int, default=2)
@click.option('--nb-task-per-gpu', type=int, default=2)
def gridc(nb_gpu, nb_task_per_gpu):
    make_run_files(nb_gpu, nb_task_per_gpu)
