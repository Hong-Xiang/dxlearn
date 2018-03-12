import unittest
from dxl.learn.core.base import Host


class TestHost(unittest.TestCase):
    def test_eq(self):
        h1 = Host('master')
        h2 = Host('master')
        self.assertEqual(h1, h2)
        h1 = Host('worker', 0)
        h2 = Host('worker', 0)
        self.assertEqual(h1, h2)
        h1 = Host('worker', 1)
        h2 = Host('worker', 0)
        self.assertNotEqual(h1, h2)
