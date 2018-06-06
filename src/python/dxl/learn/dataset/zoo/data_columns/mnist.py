from ...data_column import PyTablesColumns


class MNISTColumns(PyTablesColumns):
    def __init__(self, path_file, is_train):
        path_dataset = '/train' if is_train else '/test'
        super().__init__(path_file, path_dataset)