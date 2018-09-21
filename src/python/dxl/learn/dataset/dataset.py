import tables as tb
from dxl.learn.graph import Graph
from dxl.fs import Path
from typing import Dict
from .partitioner import Partitioner
import tensorflow as tf

RATIO_SHUFFLE_BUFFER_TO_BATCH_SIZE = 4

# from dxl.learn.function import Function, function, MapIf, NestMapOf, shape_list, To
from doufo.tensor import Tensor
from typing import Union, NamedTuple


class Dataset(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            NB_EPOCHS = 'nb_epochs'
            BATCH_SIZE = 'batch_size'
            IS_SHUFFLE = 'is_shuffle'

    def __init__(self,
                 info,
                 processing=None,
                 *,
                 nb_epochs=None,
                 batch_size=None,
                 is_shuffle=None):

        super().__init__(
            info,
            config={
                self.KEYS.CONFIG.NB_EPOCHS: nb_epochs,
                self.KEYS.CONFIG.BATCH_SIZE: batch_size,
                self.KEYS.CONFIG.IS_SHUFFLE: is_shuffle,
            })
        if processing is None:
            processing = StandardProcessing(self.config(self.KEYS.CONFIG.BATCH_SIZE),
                                            self.config(
                                                self.KEYS.CONFIG.NB_EPOCHS),
                                            self.config(self.KEYS.CONFIG.IS_SHUFFLE))
        self.processing = processing


class DatasetFromColumns(Graph):
    class KEYS(Dataset.KEYS):
        class TENSOR(Dataset.KEYS.TENSOR):
            DATA = 'data'

        class CONFIG(Graph.KEYS.CONFIG):
            NB_EPOCHS = 'nb_epochs'
            BATCH_SIZE = 'batch_size'
            IS_SHUFFLE = 'is_shuffle'

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
            config={
                self.KEYS.CONFIG.NB_EPOCHS: nb_epochs,
                self.KEYS.CONFIG.BATCH_SIZE: batch_size,
                self.KEYS.CONFIG.IS_SHUFFLE: is_shuffle,
            })

    def _make_dataset_object(self):
        return tf.data.Dataset.from_generator(
            self._columns.__iter__, self._columns.types,
            NestMapOf(tf.TensorShape)(self._columns.shapes))

    def _convert(self, v):
        result = Tensor(v)
        if self.config(self.KEYS.CONFIG.BATCH_SIZE) is not None:
            shape = result.data.shape.as_list()
            shape[0] = self.config(self.KEYS.CONFIG.BATCH_SIZE)
            if shape.count(None) == 1:
                shape[shape.index(None)] = -1
            result = Tensor(tf.reshape(result.data, shape))
        return result

    def _finalize_to_dict_of_tensors(self, dataset):
        result = dataset.make_one_shot_iterator().get_next()
        if not isinstance(result, dict):
            result = {self.KEYS.TENSOR.DATA: result}
        return {k: self._convert(v) for k, v in result.items()}

    def _process_dataset(self, dataset):
        KC = self.KEYS.CONFIG
        dataset = dataset.repeat(self.config(KC.NB_EPOCHS))
        if self.config(KC.IS_SHUFFLE):
            dataset = dataset.shuffle(self.config(KC.BATCH_SIZE) * 4)
        dataset = dataset.batch(self.config(KC.BATCH_SIZE))
        return dataset

    def kernel(self, inputs=None):
        dataset = self._make_dataset_object()
        dataset = self._process_dataset(dataset)
        self.tensors.update(self._finalize_to_dict_of_tensors(dataset))


class Batch(Function):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, x: Union[tf.data.Dataset]):
        if isinstance(x, tf.data.Dataset):
            return x.batch(self.batch_size)


class Repeat(Function):
    def __init__(self, nb_repeat=None):
        self.nb_repeat = nb_repeat

    def __call__(self, x: Union[tf.data.Dataset]):
        if isinstance(x, tf.data.Dataset):
            return x.repeat(self.nb_repeat)


class Shuffle(Function):
    def __init__(self, nb_buffer):
        self.nb_buffer = nb_buffer

    def __call__(self, x: Union[tf.data.Dataset]):
        if isinstance(x, tf.data.Dataset):
            return x.shuffle(self.nb_buffer)


class ReshapeForBatch(Function):
    def __init__(self, batch_size, axis=0):
        self.batch_size = batch_size
        self.axis = axis

    def __call__(self, x: Union[tf.data.Dataset]):
        shape = shape_list(x)
        shape[self.axis] = self.batch_size
        shape = [s if s is not None else -1 for s in shape]
        if isinstance(x, tf.Tensor):
            return tf.reshape(x, shape)
        raise TypeError("Unknown type {} of x.".format(type(x)))


class StandardProcessing(Function):
    def __init__(self, batch_size, nb_epochs, is_shuffle):
        self.f = (Repeat(nb_epochs)
                  >> MapIf(lambda _: is_shuffle, Shuffle(batch_size * 4))
                  >> Batch(batch_size))

    def __call__(self, x: Union[tf.data.Dataset]):
        if isinstance(x, tf.data.Dataset):
            return self.f(x)


class DictToTupleForDataclass(Function):
    def __init__(self, dataclass):
        self.dataclass = dataclass

    def __call__(self, d):
        return tuple([d[k] for k in self.dataclass._fields])


class TupleToDictForDataclass(Function):
    def __init__(self, dataclass):
        self.dataclass = dataclass

    def __call__(self, t):
        return {k: t[i] for i, k in enumerate(self.dataclass._fields)}


class ColumnsToTensorFlowDataset(Function):
    def __init__(self, is_with_shape=False):
        self.is_with_shape = is_with_shape

    def __call__(self, columns):
        d2t = DictToTupleForDataclass(columns.dataclass)
        if self.is_with_shape:
            return tf.data.Dataset.from_generator(
                columns.__iter__,
                NestMapOf(tf.as_dtype)(d2t(columns.dtypes)),
                NestMapOf(tf.TensorShape)(d2t(columns.dataclass.shapes())))
        else:
            return tf.data.Dataset.from_generator(
                columns.__iter__, d2t(columns.dtypes))


@function
def dataset_to_tensor(x: tf.data.Dataset):
    return x.make_one_shot_iterator().get_next()


@function
def named_tuple_to_dict(t: NamedTuple):
    return {k: t[i] for i, k in enumerate(t._fields)}


class DatasetFromColumnsV2(Dataset):
    class KEYS(Dataset.KEYS):
        class CONFIG(Dataset.KEYS.CONFIG):
            IS_WITH_SHAPE = 'is_with_shape'

        class TENSOR(Dataset.KEYS.TENSOR):
            DATA = 'data'

    def __init__(self, info, columns, processing=None, *, batch_size=None, nb_epochs=None, is_shuffle=None):
        super().__init__(info, processing, batch_size=batch_size,
                         nb_epochs=nb_epochs, is_shuffle=is_shuffle)
        self.columns = columns
        # TODO: refactor is_with_shape

    def _to_tf_tensors(self, dataset):
        result = dataset.make_one_shot_iterator().get_next()
        if not isinstance(result, dict):
            result = {self.KEYS.TENSOR.DATA: result}
        return {k: self._convert(v) for k, v in result.items()}

    def kernel(self, inputs=None):
        f = (ColumnsToTensorFlowDataset(True)
             >> self.processing
             >> dataset_to_tensor
             >> NestMapOf(ReshapeForBatch(self.config(self.KEYS.CONFIG.BATCH_SIZE)))
             >> NestMapOf(To(Tensor))
             >> TupleToDictForDataclass(self.columns.dataclass))
        self.tensors.update(f(self.columns))


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
