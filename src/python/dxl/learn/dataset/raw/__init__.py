"""
Module of *raw* datasets, thus only handles reading dataset files (usually .h5 or .tfrecord),
parse records/examples into tensors in dict-like collection, no normalization in principle.

However some properties may be provided, like mean or standard deviation.
"""