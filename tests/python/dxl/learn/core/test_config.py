import unittest
import dxl.learn.core.config as dlcc


class TestConfigurableWithName(unittest.TestCase):
    def test_basic(self):
        c = dlcc.ConfigurableWithName('x', {'a': 1, 'b': 2})
        self.assertEqual(c.config('a'), 1)
