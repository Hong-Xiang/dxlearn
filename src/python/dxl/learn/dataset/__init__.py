"""
This package is designed to simplify process of generating high performance (e.g. I/O) dataset objects,
thus tf.data compatible objects currently.

Currently only one model of dataset is implemented.

A Dataset is a dict of ndarray or dict. Thus in general, a dataset is a dict of ndarrays which possible special keys
like 'train/x'.
All ndarray are at least one dimensional, thus `len(data.shape)` > 1, and the first dimension is dimension of samples.

The first problem of this package solved is unified access to data stored in filesystem.
We are targeting to provide support for

1. HDF5
2. NPY, NPZcdtw`
3. MAT

A unified FSDatasetSpec is used to describe these datasets. It works like a normal dict, with one or more <name>,
indicating column features. <name> can be path-like, i.e. 'train/x'

A typical example is
{
'x': {
'path_file': '~/Data/mnist.h5',
'path_dataset': 'train/x',
}
'y': {
'path_file': '~/Data/mnist.h5',
'path_dataset': 'train/y'
}
}
'path_file' can be omitted for some column, at these cases, if there is a `path_file` filed in root dict, it will be used.
`path_dataset' can be omitted, as this case, the name of column will be used.
"""
from .data_column import (ListColumns, PyTablesColumns, DataColumns,
                          HDF5DataColumns, NPYDataColumns, NPZDataColumns,
                          RangeColumns, DataColumnsPartition)
from .partitioner import CrossValidatePartitioner, Train80Partitioner
# from .dataset import Dataset, DatasetFromColumns, DatasetFromColumnsV2
# from .api import get_dataset
