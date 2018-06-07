"""
Dataset partition utilities.

Class Partition is a index provider, thus provide index of next sample in ndarray.

For example, for a MNIST dataset stored in a HDF5 file:

dataset:
    train:
        x: [1000, 28, 28, 1]
        y: [1000, 1]
    test:
        x: [100, 28, 28, 1]
        y: [100, 1]

If we create a Partition by:

```Python
dataset = some_dataset_loader()
>>> dataset.capacity('train')
>>> 1000
>>> p = Partition(range(dataset.capacity('train')))
>>> next(p)
0
>>> next(p)
1
# ...
>>> next(p)
999
>>> next(p)
0
```



next(p['train']) # 0
next(p['train']) # 1
next(p['test']) # 0
...
next(p['test']) # 99
next(p['test']) # 0 
```

methods:

```
p.capacity(partition_name)
```
```
>>> p.capacity('train')
>>> 100
```

Partition maybe not useful when dataset was already seperated, but for cases:
dataset:
    inputs: [1100, 28, 28, 1]
    labels: [1100, 1]
thus, train and test dataset was not seperated yes, it would be useful.

```Python
dataset = some_dataset_loader()
p = Partition(partitions={'train': range(1000), 'test': range(1000, 1100)})
next(p['train']) # 0
next(p['train']) # 1
next(p['test']) # 1000
```
"""
from collections import UserDict
from typing import Dict, Iterable

import numpy as np
from .data_column import DataColumns, DataColumnsWithGetItem


class Partitioner:
    def _get_original_indices(self, data_column):
        return range(data_column.capacity)

    def _get_valid_indices(self, indicies):
        return indicies

    def partition(self, data_column: Iterable) -> Iterable:
        def valid_index_generator(data_column, indices):
            for i in indices:
                yield data_column[i]

        return valid_index_generator(
            data_column,
            self._get_valid_indices(self._get_original_indices(data_column)))

    def get_capacity(self, data_columns):
        raise NotImplementedError


class CrossValidatePartitioner(Partitioner):
    def __init__(self, nb_blocks, in_blocks):
        super().__init__()
        self._nb_blocks = nb_blocks
        if isinstance(in_blocks, int):
            in_blocks = [in_blocks]
        self._in_blocks = in_blocks

    def _get_valid_indices(self, indices):
        result = []
        len_block = len(indices) // self._nb_blocks
        for b in self._in_blocks:
            result += [
                indices[i] for i in range(b * len_block, (b + 1) * len_block)
            ]
        return tuple(result)

    def get_capacity(self, data_columns):
        len_block = data_columns.capacity // self._nb_blocks
        return len_block * len(self._in_blocks)


class Train80Partitioner(CrossValidatePartitioner):
    def __init__(self, is_train):
        nb_blocks, nb_train = 10, 8
        if is_train:
            in_blocks = range(nb_train)
        else:
            in_blocks = range(nb_train, nb_blocks)
        super().__init__(nb_blocks, in_blocks)
