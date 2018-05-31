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
from typing import Iterable, Dict
from enum import Enum


class PartGenerator:
    def __init__(self, ids, nb_epochs=None):        
        if nb_epochs is None:
            nb_epochs = -1
        self.nb_epochs = nb_epochs
        self.ids = ids
        self._g = self._make_generator()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._g)

    def _make_generator(self):
        current_epoch = 0
        while self.nb_epochs == -1 or current_epoch < self.nb_epochs:
            for i in self.ids:
                yield i
            current_epoch += 1


class Partition:
    class KEYS:
        TRAIN = 'train'
        VALIDATE = 'validate'
        DEVELOP = 'develop'
        TEST = 'test'
        EVALUATE = 'evaluate'

    def __init__(self, partitions:Dict[str, Iterable], nb_epochs=None):
        if nb_epochs is None:
            nb_epochs = -1
        self.nb_epochs = nb_epochs
        self.partitions = partitions
        self.iterators = {
            k: self.make_iterator(v)
            for k, v in self.partitions.items()
        }

    def capacity_of(self, name):
        if self.nb_epochs == -1:
            return float('inf')
        else:
            return len(self.partitions[name])

    def make_iterator(self, ids):
        current_epoch = 0
        while self.nb_epochs == -1 or current_epoch < self.nb_epochs:
            for i in ids:
                yield i
            current_epoch += 1

    def __getitem__(self, name):
        return self.iterators[name]


    # def ids(self, name) -> Iterable[int]:
    #     raise NotImplementedError

    # def partitions(self) -> Iterable[str]:
    #     raise NotImplementedError

class CrossValidate(Partition):
    def __init__(self,
                 cross: Dict[str, Iterable],
                 capacity=None,
                 nb_blocks=None,
                 nb_epochs=None):
        self.cross = {}
        nb_id_ablock, _ = divmod(capacity, nb_blocks)
        for k, v in cross.items():
            index = []
            for id_block in v:
                start = id_block * nb_id_ablock
                end = start + nb_id_ablock
                index.extend(list(range(start, end)))
            self.cross.update({k, index})
        super().__init__(self.cross, nb_epochs=nb_epochs)


class Train80(Partition):
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


class ExplicitIds(Partition):
    def __init__(self, ids_dict: dict):
        self._ids_dict = ids_dict

    def ids(self, name) -> Iterable[int]:
        return tuple(self._ids_dict[name])

    def partitions(self):
        return tuple(self._ids_dict.keys())
