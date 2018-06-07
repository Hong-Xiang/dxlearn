import tables as tb
from ..core import Graph, Tensor
from dxl.fs import Path
from typing import Dict
from .partitioner import Partitioner
import tensorflow as tf


class Dataset(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            NB_EPOCHS = 'nb_epochs'
            BATCH_SIZE = 'batch_size'
            IS_SHUFFLE = 'is_shuffle'

    def __init__(self,
                 info,
                 *,
                 nb_epochs=None,
                 batch_size=None,
                 is_shuffle=None,
                 config=None):

        super().__init__(
            name,
            config=self._parse_input_config({
                self.KEYS.CONFIG.NB_EPOCHS:
                nb_epochs,
                self.KEYS.CONFIG.BATCHSIZE:
                batch_size,
                self.KEYS.CONFIG.IS_SHUFFLE:
                is_shuffle
            }))

    def _process_dataset(self, dataset):
        KC = self.KEYS.CONFIG
        dataset = dataset.repeat(self.config(KC.NB_EPOCHS))
        if self.config(KC.IS_SHUFFLE):
            dataset = dataset.shuffle(self.config(KC.BATCH_SIZE) * 4)
        dataset = dataset.batch(self.config(KC.BATCH_SIZE))
        return dataset


class DatasetFromColumns(Dataset):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            DATA = 'data'

    def __init__(self,
                 info,
                 columns,
                 *,
                 nb_epochs=None,
                 batch_size=None,
                 is_shuffle=None,
                 config=None):
        self._columns = columns
        super().__init__(
            info,
            nb_epochs=nb_epochs,
            batch_size=batch_size,
            is_shuffle=is_shuffle,
            config=config)

    def _make_dataset_object(self):
        return tf.data.Dataset.from_generator(self._columns.__iter__)

    def _make_dataset_tensor(self, dataset):
        return Tensor(dataset.make_one_shot_iterator().get_next())

    def _kernel(self):
        dataset = self._make_dataset_object()
        dataset = self._process_dataset(dataset)
        self.tensors[self.KEYS.TENSOR.DATA] = self._make_dataset_tensor(
            dataset)


# class HDF5Dataset(Dataset):
#     '''Default pytables
#     '''
#     class KEYS(Dataset.KEYS):
#         class TENSOR:
#             pass
#         class CONFIG:
#             IN_MEMORY = 'in_memory'
#             FIELD = 'field'
#         class CMD:
#             ITER = 'hand=h5.{}.iterrows'

#     def __init__(self, name, config, info=None):
#         super().__init__(
#             name=name,
#             config=config,
#             info=info)

#     def loader(self, name):
#         with tb.open_file(name, mode="r") as h5:
#             handels = {}
#             field = self.config(self.KEYS.CONFIG.FIELD)
#             for k, v in field.items():
#                 hand = []
#                 cmd = 'hand=h5.{}.iterrows'.format(v)
#                 exec(cmd)

#                 if self.config(self.KEYS.CONFIG.IN_MEMORY):
#                     hand = []
#                     cmd = 'hand=[x[{}] for x in h5.{}.iterow]'
#                 else:

#                 handels.update({k : hand})

#             return handels

#     def pre_processing(self, handel):
#         pass

# class FileDataset(Dataset):
#     pass

# class NpyDataset(Dataset):
#     pass
