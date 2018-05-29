from dxl.learn.backend import current_backend, TensorFlow
import os
from pathlib import Path
import numpy as np


class TestCase(current_backend().TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        if isinstance(current_backend(), TensorFlow):
            current_backend().unbox().reset_default_graph()

    @property
    def test_resource_path(self):
        return Path(os.getenv('DEV_DXLEARN_TEST_RESOURCE_PATH'))

    def assertFloatArrayEqual(self, first, second, msg):
        return np.testing.assert_array_almost_equal(
            np.array(first), np.array(second), msg)
    
