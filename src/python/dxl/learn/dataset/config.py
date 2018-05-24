import os
from dxpy.filesystem import Path


def _default_dataset_path():
    return str(Path(os.environ.get('HOME')) / 'Datas')


config = {
    'dataset_root_path': os.environ.get('PATHS_DATASET', _default_dataset_path()),
    'PATH_DATASETS': os.environ.get('PATHS_DATASET', _default_dataset_path())}
