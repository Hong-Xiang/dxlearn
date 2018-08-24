"""
Target data source:

In-memory tensor: Dict[np.ndarray]

Tensor generator: Iterator which return Dict[np.ndarray]
"""
from typing import Dict
from doufo.tensor import Tensor
import tensorflow as tf

class Dataset:
    """
    """
    pass


def standard_processing(spec, dataset:Dataset) -> Dict[Tensor]:
    pass

def batch(batch_size, dataset:Dataset)->Dataset:
    return dataset.fmap(lambda d: d.batch(batch_size))

def to_tensors(dataset:Dataset)->Dict[Tensor]:
    """
    Convert dataset to Tensor.
    :param dataset: Dataset, currently it's only a wrapper of tf.data.Dataset
    :return: dict of Tensors, currently a doufo.Tensor wrapped tf.Tensor.
    """
    with tf.variable_scope('finalize_dataset_to_dict_of_tensors'):
        result = dataset.unbox().make_one_shot_iterator().get_next()
        return {k: Tensor(v) for k, v in result.items()}
    

