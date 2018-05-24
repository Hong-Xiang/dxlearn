import tensorflow as tf


class SupportedDatasetNames:
    AnalyticalPhantoms = 'analytical_phantoms'


def pet_image_super_resolution_dataset(dataset_name: str,
                                       image_type: str,
                                       batch_size: int,
                                       nb_down_sample: int,
                                       target_shape: list,
                                       *,
                                       name: str='dataset',
                                       path_dataset: "str|None"=None):
    """
    Args:
        -   dataset_name: one of the following names:
            1.  analytical_phantoms
            2.  monte_carlo_phantoms [TODO: Impl]
            3.  mice [TODO: Imple]
            4.  mnist

        -   image_type: 'sinogram' or 'image'
        -   batch_size
    Returns:
        a `Graph` object, which has several nodes:
    Raises:
    """
    from ..petsino import PhantomSinograms
    from ...model.normalizer.normalizer import FixWhite, ReduceSum
    from ..super_resolution import SuperResolutionDataset
    from ...model.tensor import ShapeEnsurer
    from ...config import config
    if dataset_name is None:
        dataset_name = 'analytical_phantoms'
    normalizer_configs = {
        'analytical_phantoms': {'mean': 4.88, 'std': 4.68}
    }
    config_origin = {}
    config_normalizer = normalizer_configs[dataset_name]
    config['dataset'] = {
        'origin': config_origin,
        'fix_white': config_normalizer
    }
    with tf.name_scope('{img_type}_dataset'.format(img_type=image_type)):
        dataset_origin = PhantomSinograms(name='dataset/origin',
                                          batch_size=batch_size,
                                          fields=image_type)
        dataset_summed = ReduceSum('dataset/reduce_sum',
                                   dataset_origin[image_type],
                                   fixed_summation_value=1e6).as_tensor()

        dataset = FixWhite(name='dataset/fix_white',
                           inputs=dataset_summed)()
        dataset = tf.random_crop(dataset,
                                 [batch_size] + list(target_shape) + [1])
        dataset = SuperResolutionDataset('dataset/super_resolution',
                                         lambda: {'image': dataset},
                                         input_key='image',
                                         nb_down_sample=3)
    return dataset
