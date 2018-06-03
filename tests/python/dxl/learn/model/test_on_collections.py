from dxl.learn.test import TestCase
from dxl.learn.core import Constant
from dxl.learn.model import Summation


class TestSummation(TestCase):
    def test_directly_constructed(self):
        x = Constant(1.0)
        y = Constant(2.0)
        s = Summation('summation', [x, y])
        v = s()
        with self.test_session() as sess:
            pass
