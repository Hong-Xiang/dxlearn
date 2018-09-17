from pathlib import Path
from dxl.core.config import ConfigProxy, CNode, CView, Configuration
import os
import platform

home_str = 'HOME' if (platform.platform()[0] == 'L') else 'HOMEPATH'

datasets_configs = {
    'dataset_root_path': os.environ.get('PATHS_DATASET', str(Path(os.environ.get(home_str)) / 'Datas')),
    'path': '/home/hongxwing/Datas/',
    'analytical_phantom_sinogram': {
        'path': '/home/hongxwing/Datas/Phantom',
    },
    'apssr': {
        'image_type': 'sinogram',
        'target_shape': [320, 320],
        'super_resolution': {
            'nb_down_sample': 3
        },
    },
    'mice_sinograms': {
        'path': str(Path(os.environ.get(home_str)) / 'Datas' / 'Mice')
    }
}

config = {
    'train': {
        'summary_freq': 60,
        'ckpt_path': './save'
    },
    'datasets': datasets_configs
}

DEFAULT_CONFIGURATION_NAME = 'DXLEARN_CONFIGURATRION'


def config_with_name(name):
    if ConfigProxy().get(DEFAULT_CONFIGURATION_NAME) is None:
        ConfigProxy()[DEFAULT_CONFIGURATION_NAME] = Configuration(CNode({}))
    return ConfigProxy().get_view(DEFAULT_CONFIGURATION_NAME, name)


def clear_config():
    # del ConfigProxy()[DEFAULT_CONFIGURATION_NAME]
    ConfigProxy.reset()
