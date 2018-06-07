from .data_column import DataColumns


class DataLoader:
    def __init__(self, columns, partition):
        self._columns = columns
        self._partition = partition

    @property
    def columns(self):
        return self._columns.columns

    def _calculate_capacity(self):
        return self._partition.capacity
