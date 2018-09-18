import pytest

from dxl.learn.test import TestCase
from dxl.learn.core import Constant
# from dxl.learn.model import Summation

@pytest.mark.skip()
class TestSummation(TestCase):
    def test_directly_constructed(self):
        x = Constant(1.0, 'x')
        y = Constant(2.0, 'y')
        s = Summation('summation', [x, y])
        v = s()
        with self.test_session() as sess:
            assert sess.run(v) == 3.0

    def test_summation_of_same_tensor(self):
        x = Constant(1.0, 'x')
        s = Summation('summation', [x, x, x])
        v = s()
        with self.test_session() as sess:
            assert sess.run(v) == 3.0

    def test_construct_with_none_input(self):
        x = Constant(1.0, 'x')
        s = Summation('summation')
        v = s([x] * 3)
        with self.test_session() as sess:
            assert sess.run(v) == 3.0

    def test_inputs(self):
        x = Constant(1.0, 'x')
        s = Summation('summation', [x] * 3)
        assert s.tensors['input'] == [x] * 3
