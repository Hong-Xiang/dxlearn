from dxl.learn import Model
from dxl.learn.function import OneHot


class IncidentEstimation(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            HITS = 'hits'
            FIRST_HIT_INDEX = 'first_hit_index'

    def __init__(self, hits, first_hit_index, nb_max_hits, info):
        super().__init__(
            inputs={
                self.KEYS.TENSOR.HITS: hits,
                self.KEYS.first_hit_index: first_hit_index,
            }
        )

    def kernel(self, inputs):
        hits = inputs[self.KEYS.TENSOR.HITS]
        if self.KEYS.TENSOR.FIRST_HIT_INDEX in inputs:
            label = OneHot()
