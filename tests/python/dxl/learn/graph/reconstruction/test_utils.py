import unittest
from dxl.learn.graph.reconstruction import utils


class TestDataInfo(unittest.TestCase):
  def test_lor_file(self):
    _, data_info = utils.load_reconstruction_configs()
    self.assertEqual(data_info.lor_file('x', 0), './debug/xlors.npy')
    self.assertEqual(data_info.lor_file('y', 0), './debug/ylors.npy')
    self.assertEqual(data_info.lor_file('z', 0), './debug/zlors.npy')

  def test_lor_shapes(self):
    _, data_info = utils.load_reconstruction_configs()
    self.assertEqual(data_info.lor_shape('x', 0), [100, 6])
    self.assertEqual(data_info.lor_shape('y', 0), [200, 6])
    self.assertEqual(data_info.lor_shape('z', 0), [300, 6])