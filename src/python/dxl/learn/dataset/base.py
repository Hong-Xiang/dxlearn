from ..core import Graph

"""
A dataset should handle:
1.  Loading raw data (ndarray or nested ndarrays)
2.  Partition into different part
3.  Processing to tf.Tensor or Tensor as interface
"""

class Dataset(Graph):
    def __init__(self,
                 name,
                 tensors=None,
                 subgraphs=None,
                 info=None,
                 config=None,
                 partition=None,
                 ):
        super.__init__(name, tensors, subgraphs, info, config)

    @property
    def capacity(self):
        pass


class HDF5Dataset(Graph):
    """
    """

    def __init__(self, name, ,):
        pass


