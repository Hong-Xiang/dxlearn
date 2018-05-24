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
p = Partition(partitions={'train': range(dataset.capacity('train')), 'test': range(dataset.capacity('test'))})
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
from typing import Iterable
from enum import Enum


class Partition:
    class KEYS:
        TRAIN = 'train'
        VALIDATE = 'validate'
        DEVELOP = 'develop'
        TEST = 'test'
        EVALUATE = 'evaluate'

    def __init__(self, partitions, nb_epochs=None):
        if nb_epochs is None:
            nb_epochs = -1
        self.nb_epochs = nb_epochs
        self.partitions = partitions
        self.iterators = {
            k: self.make_iterator(v)
            for k, v in self.partitions.items()
        }

    def capacity_of(self, name):
        return len(self.partitions[name])

    def make_iterator(self, ids):
        current_epoch = 0
        while nb_epochs == -1 or current_epoch < nb_epochs:
            for i in ids:
                yield i
            current_epoch += 1

    def __getitem__(self, name):
        return self.iterators[name]

    # def ids(self, name) -> Iterable[int]:
    #     raise NotImplementedError

    # def partitions(self) -> Iterable[str]:
    #     raise NotImplementedError


class CrossValidate(DatasetPartition):
    def __init__(self,
                 dataset,
                 *,
                 nb_partitions=None,
                 idx_test=None,
                 is_shuffle=None,
                 nb_epochs=None):
        super().__init__(dataset, is_shuffle=is_shuffle, nb_epochs=nb_epochs)
        self.nb_partitions = nb_partitions
        self.idx_test = idx_test
        self.capacity = self.capacity_of(dataset)


class Train80(DatasetPartition):
    def __init__(self, dataset, partition=None):
        """
        `dataset` Dataset object, with dataset.capacity property.
        `partition` str, name of partition
        """
        nb_train = int(dataset.size * 0.8)
        nb_test = dataset.size - nb_train
        self._ids_train = range(nb_train)
        self._ids_test = range(nb_train, dataset.size)

    def ids(self, name) -> Iterable[int]:
        return {'train': self._ids_train, 'test': self._ids_test}[name]

    def partitions(self):
        return ('train', 'test')


class ExplicitIds(DatasetPartition):
    def __init__(self, ids_dict: dict):
        self._ids_dict = ids_dict

    def ids(self, name) -> Iterable[int]:
        return tuple(self._ids_dict[name])

    def partitions(self):
        return tuple(self._ids_dict.keys())
