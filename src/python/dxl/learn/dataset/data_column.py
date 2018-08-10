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


from dxl.data import ColumnsWithIndex
from typing import NamedTuple, Optional, Dict
from pathlib import Path


class PyTablesColumnsV2(ColumnsWithIndex):
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
