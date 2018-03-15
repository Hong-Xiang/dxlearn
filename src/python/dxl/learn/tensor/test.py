from ..core import Tensor


class TensorTest(Tensor):
    def add_one(self):
        with self.graph_info.variable_scope():
            return TensorTest(self.data + 1, self.data_info, self.graph_info.update(name=None))
