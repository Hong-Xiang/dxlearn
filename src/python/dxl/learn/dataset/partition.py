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


class Partition:
    def __init__(self, indices, nb_epochs=None):
        self.indices = indices
        if nb_epochs is None:
            nb_epochs = np.inf
        self.nb_epochs = nb_epochs
        self._gen = None

    def _make_index_generator(self):
        current_epoch = 0
        while current_epoch < self.nb_epochs:
            for i in self.indices:
                yield i
            current_epoch += 1

    @property
    def capacity(self):
        return len(self.indices) * self.nb_epochs

    def __next__(self):
        if self._gen is None:
            self._gen = self._make_index_generator()
        return next(self._gen)

    def __iter__(self):
        return self


class CrossValidatePartition(Partition):
    def __init__(self, indices: Iterable, nb_blocks, in_blocks,
                 nb_epochs=None):
        super().__init__(
            self._get_valid_indices(indices, nb_blocks, in_blocks),
            nb_epochs=nb_epochs)

    def _get_valid_indices(self, indices, nb_blocks, in_blocks):
        if isinstance(in_blocks, int):
            in_blocks = [in_blocks]
        result = []
        len_block = len(indices) // nb_blocks
        for b in in_blocks:
            result += [
                indices[i] for i in range(b * len_block, (b + 1) * len_block)
            ]
        return tuple(result)


class Train80Partition(CrossValidatePartition):
    def __init__(self, indicies, is_train, nb_epochs=None):
        nb_blocks, nb_train = 10, 8
        if is_train:
            in_blocks = range(nb_train)
        else:
            in_blocks = range(nb_train, nb_blocks)
        super().__init__(indicies, nb_blocks, in_blocks, nb_epochs)


class Partitions(UserDict):
    class K:
        TRAIN = 'train'
        VALIDATE = 'validate'
        DEVELOP = 'develop'
        TEST = 'test'
        EVALUATE = 'evaluate'

    def __init__(self,
                 partitions: Dict[str, Partition] = None,
                 *,
                 train=None,
                 test=None,
                 develop=None,
                 validate=None,
                 evaluate=None):
        if partitions is None:
            partitions = {}
        partitions.update(
            self.__dict_filter_out_none({
                self.K.TRAIN: train,
                self.K.TEST: test,
                self.K.DEVELOP: develop,
                self.K.VALIDATE: validate,
                self.K.EVALUATE: evaluate,
            }))
        self.data = partitions

    @classmethod
    @property
    def KEYS(self):
        return self.K

    def __dict_filter_out_none(self, dct):
        result = {}
        for k, v in dct.items():
            if v is not None:
                result[k] = v
        return result

    def capacity_of(self, name) -> int:
        return self.data[name].capacity()
