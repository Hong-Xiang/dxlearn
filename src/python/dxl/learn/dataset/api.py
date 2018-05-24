from dxpy.configs import configurable
from dxpy.learn.config import config
from dxpy.core.path import Path


@configurable(config, with_name=True)
def get_dataset(name, dataset_cls_name, dataset_config_name=None):
    """
    Args:
        name: config name
    """
    if dataset_config_name is None:
        dataset_config_name = name
    if dataset_cls_name == 'sraps':
        from .zoo.sraps import AnalyticalPhantomSinogramDatasetForSuperResolution 
        return AnalyticalPhantomSinogramDatasetForSuperResolution(name=dataset_config_name)
    elif dataset_cls_name == 'sin':
        from .zoo.sin import SinDataset
        return SinDataset(name=dataset_config_name)        
    elif dataset_cls_name == 'srext':
        from .zoo.srext import SuperResolutionExternalDataset
        return SuperResolutionExternalDataset(name=dataset_config_name)
    else:
        raise ValueError(
            'Unknown dataset_name (class name) {}.'.format(dataset_cls_name))
