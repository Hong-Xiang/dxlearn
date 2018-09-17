"""
DataColumns, a representation of table-like data.
"""
import h5py
import tables as tb
import numpy as np
from typing import Dict, Iterable
from dxl.fs import Path
import tensorflow as tf
from dxl.data.io import load_npz
from typing import Tuple, TypeVar


class DataColumns:
    """
    """

    def __init__(self, data):
        self.data = self._process(data)
        self._capacity_cache = None
        self._iterator = None

    def _process(self, data):
        return data

    @property
    def columns(self):
        return self.data.keys()

    def _calculate_capacity(self):
        raise NotImplementedError

    @property
    def capacity(self):
        if self._capacity_cache is not None:
            return self._capacity_cache
        else:
            return self._calculate_capacity()

    @property
    def shapes(self):
        raise NotImplementedError

    @property
    def types(self):
        raise NotImplementedError

    def _make_iterator(self):
        raise NotImplementedError

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self)
        return next(self._iterator)

    def __iter__(self):
        return self._make_iterator()


class DataColumnsPartition(DataColumns):
    def __init__(self, data: DataColumns, partitioner):
        super().__init__(data)
        self._partitioner = partitioner

    def _make_iterator(self):
        return self._partitioner.partition(self.data)

    def _calculate_capacity(self):
        return self._partitioner.get_capacity(self.data)

    @property
    def shapes(self):
        return self.data.shapes

    @property
    def types(self):
        return self.data.types


class DataColumnsWithGetItem(DataColumns):
    def _make_iterator(self):
        def it():
            for i in range(self.capacity):
                yield self.__getitem__(i)

        return it()

    def __getitem__(self, i):
        raise NotImplementedError


class RangeColumns(DataColumnsWithGetItem):
    def __init__(self, nb_samples):
        super().__init__(nb_samples)

    def _calculate_capacity(self):
        return self.data

    @property
    def shapes(self):
        return tuple()

    @property
    def types(self):
        return tf.int32

    def __getitem__(self, i):
        return i


class JointDataColumns(DataColumns):
    def __init__(self, data, name_map):
        super().__init__(data)
        self.name_map = name_map

    def __getitem__(self, i):
        result = {}
        for d, m in zip(self.data, self.name_map):
            r = d[i]
            for k, v in r.items():
                if k in m:
                    result[self.name_map[k]] = v
        return result

    def _calculate_capacity(self):
        capacities = set(d._calculate_capacity() for d in self.data)
        if len(capacities) != 1:
            raise ValueError("Inconsistant capacities {}.".format(capacities))
        return capacities[0]


class NDArrayColumns(DataColumnsWithGetItem):
    def _calculate_capacity(self):
        result = None
        for k in self.columns:
            current_capacity = self.data[k].shape[0]
            if result is None:
                result = current_capacity
            else:
                if result != current_capacity:
                    raise ValueError(
                        "Capacity of {} is not equal to previous.".format(k))
        return result

    def __getitem__(self, i):
        result = {}
        for k in self.columns:
            result[k] = self.data[k][i, ...]


class ListColumns(DataColumnsWithGetItem):
    def _calculate_capacity(self):
        result = None
        for k in self.columns:
            current_capacity = len(self.data[k])
            if result is None:
                result = current_capacity
            else:
                if result != current_capacity:
                    raise ValueError(
                        "Capacity of {} is not equal to previous.".format(k))
        return result

    def __getitem__(self, i):
        result = {}
        for k in self.columns:
            result[k] = self.data[k][i]


class HDF5DataColumns(NDArrayColumns):
    def _process(self, data):
        if isinstance(data, (str, Path)):
            data = h5py.File(data)
        return data

    def close(self):
        self.data.close()


class NPYDataColumns(NDArrayColumns):
    class K:
        DATA = 'data'

    def _process(self, data):
        return np.load(data)


class NPZDataColumns(NDArrayColumns):
    def _process(self, data):
        return load_npz(data)


class PyTablesColumns(DataColumnsWithGetItem):
    def __init__(self, path_file, path_dataset):
        super().__init__((path_file, path_dataset))

    def _process(self, data):
        path_file, path_dataset = data
        self._file = tb.open_file(str(path_file))
        self._node = self._file.get_node(path_dataset)

    def __getitem__(self, i):
        result = {}
        data = self._node[i]
        for k in self.columns:
            result[k] = np.array(data[k])
        return result

    @property
    def columns(self):
        return tuple(self._node.colnames)

    @property
    def types(self):
        result = {}
        coltypes = self._node.coltypes
        for k, v in coltypes.items():
            result.update({k: tf.as_dtype(v)})
            # result.update({k: tf.float32})
        return result

    @property
    def shapes(self):
        result = {}
        coldescrs = self._node.coldescrs
        for k, v in coldescrs.items():
            result.update({k: v.shape})
        return result

    def _calculate_capacity(self):
        return self._node.shape[0]

    def close(self):
        self._file.close()


class ColumnsWithIndex:
    pass


# from dxl.data import ColumnsWithIndex
from typing import NamedTuple, Optional, Dict
from pathlib import Path
from doufo._dataclass import dataclass
import tables as tb
import attr
import typing
import tensorflow as tf
from enum import Enum


@attr.s(frozen=False, auto_attribs=True, slots=True)
class PytableData:
    class KEYS:
        class ATTR:
            PATH = 'path'
            FILE = 'file'

        class MARK:
            TABLE = 'table'
            DATASET = 'dataset'

        class MODE:
            TENSOR = 'Tensor'

    """ pytablereader attribute defination. """

    file_path: str = attr.ib()
    file: typing.Any = attr.ib(default=None)
    table: dict = attr.ib(default={})
    file_dataset: dict = attr.ib(default={})
    mode: str = attr.ib(default='r')
    type_mapper: dict = attr.ib(default={
        np.dtype('float16'): tf.float16,
        np.dtype('float32'): tf.float32,
        np.dtype('float64'): tf.float64,
        np.dtype('int8'): tf.int8,
        np.dtype('int16'): tf.int16,
        np.dtype('int32'): tf.int32,
        np.dtype('int64'): tf.int64,
    })


class PytableReader(PytableData):

    def check(self, attr: str, mark=None):
        if attr == self.KEYS.ATTR.PATH:
            if self.file_path is None:
                raise ValueError('path is empty.')
        if attr == self.KEYS.ATTR.FILE:
            if self.file is None:
                raise ValueError('file is empty.')
        if mark == self.KEYS.MARK.TABLE:
            if attr not in self.table.keys():
                raise ValueError('table not found.')
        if mark == self.KEYS.MARK.DATASET:
            # print(attr, self.file_dataset)
            if attr not in self.file_dataset.keys():
                raise ValueError('dataset not found.')

    def open(self, file_path: str):
        return tb.open_file(str(file_path), mode=self.mode)

    def __enter__(self):
        self.file = self.open(str(self.file_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def get_h5_to_table(self, table_dir: str, table_name=None):
        self.check(self.KEYS.ATTR.FILE)
        if table_dir is None and table_name is None:
            raise ValueError('no table to locate.')
        temp_table = self.file.get_node(table_dir, name=table_name)
        self.add_table_to_dic(temp_table.name, temp_table)
        return self.retrieve_table(temp_table.name)

    def retrieve_table(self, table_name):
        return self.table[table_name]

    def add_table_to_dic(self, table_name, table):
        self.table[table_name] = table

    def make_iterator(self, table):
        self.check(table.name, self.KEYS.MARK.TABLE)
        it = table.iterrows()
        col_name = table.colnames
        col_num = len(col_name)

        def iterator():
            for row in it:
                res = []
                for col in range(col_num):
                    res.append(row[col])
                yield tuple(res)

        return iterator

    def map_to_tf_type(self, ntype: np.dtype):
        return self.type_mapper[ntype]

    def get_type_and_shape(self, table):
        res_types = []
        res_shapes = []
        for name in table.colnames:
            res_type = table.col(name)[0].dtype
            res_shape = table.col(name)[0].shape
            res_shape = tf.TensorShape(list(res_shape))
            res_type = self.map_to_tf_type(res_type)
            # print(res_type)
            res_types.append(res_type)
            res_shapes.append(res_shape)
        return tuple(res_types), tuple(res_shapes)

    def to_dataset(self, table_name: str):
        self.check(table_name, self.KEYS.MARK.TABLE)
        table = self.retrieve_table(table_name)
        it = self.make_iterator(table)
        table_type, table_shape = self.get_type_and_shape(table)
        dataset = tf.data.Dataset.from_generator(it, table_type, table_shape)
        self.add_dataset_to_dic(table_name, dataset)
        return self.retrieve_dataset(table_name)

    def add_dataset_to_dic(self, dataset_name, dataset):
        self.file_dataset[dataset_name] = dataset

    def retrieve_dataset(self, dataset_name):
        self.check(dataset_name, self.KEYS.MARK.DATASET)
        return self.file_dataset[dataset_name]

    def process_dataset(self, dataset, repeat=None, is_shuffle=False, batch_size=1):
        if is_shuffle == True:
            return dataset.repeat(repeat).shuffle(buffer_size=1000).batch(batch_size)
        else:
            return dataset.repeat(repeat).batch(batch_size)

    def get_data(self, data_set, iterator=None, mode='Tensor'):
        if mode == self.KEYS.MODE.TENSOR:
            return self.to_tensor(data_set, iterator=iterator)
        # reserve for other data format.
        else:
            pass

    def to_tensor(self, data_set, iterator=None):
        it = iterator or data_set.make_one_shot_iterator()
        return it.get_next()


class PyTablesColumnsV2():
    def __init__(self, path: Path, dataclass: NamedTuple, key_map=Optional[Dict[str, str]]):
        super().__init__(dataclass)
        self.path = path
        self.key_map = key_map
        self.file = None

    def __enter__(self):
        self.file = self.open()
        return self.file

    def __exit__(self, type, value, tb):
        self.file.close()

    def open(self):
        return h5py.File(self.path, 'r')

    def close(self):
        return self.file.close()

    @property
    def capacity(self):
        if self.file is None:
            raise TypeError("PyTable not initialied yet.")
