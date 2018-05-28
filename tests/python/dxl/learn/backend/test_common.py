from dxl.learn.backend import current_backend, TensorFlow
import unittest


class TestBackendManager(unittest.TestCase):
    def test_default_current_backend(self):
        self.assertIsInstance(current_backend(), TensorFlow)
