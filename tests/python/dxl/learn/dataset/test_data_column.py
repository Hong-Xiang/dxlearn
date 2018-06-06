from dxl.learn.dataset import ListColumns
import unittest


class TestListColumns(unittest.TestCase):
    def test_construct(self):
        nb_samples = 100
        x = range(nb_samples)
        y = [(x_**2, x_ + 10) for x_ in x]
        c = ListColumns({'x': x, 'y': y})
        assert c.capacity == 100
