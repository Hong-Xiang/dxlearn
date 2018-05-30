import os
from pathlib import Path
import numpy as np
import uuid
from dxl.learn.backend import current_backend
from dxl.learn.utils.test_utils import name_str_without_colon_and_index, get_object_name_str


class TestCase(current_backend().TestCase()):
    def setUp(self):
        pass

    def tearDown(self):
        from dxl.learn.backend import current_backend, TensorFlow
        from dxl.learn.core.config import clear_config
        if isinstance(current_backend(), TensorFlow):
            current_backend().unbox().reset_default_graph()
        clear_config()

    def make_dummy_tensor(self, info=None):
        from dxl.learn.core import Constant
        if info is None:
            info = str(uuid.uuid4())
        return Constant(graph_info=info, data=0.0)

    @property
    def resource_path(self):
        return Path(os.getenv('DEV_DXLEARN_TEST_RESOURCE_PATH'))

    def assertFloatArrayEqual(self, first, second, msg):
        return np.testing.assert_array_almost_equal(
            np.array(first), np.array(second), msg)

    def assertNameEqual(self, first, second, with_strip_colon_and_index=True):
        names = map(get_object_name_str, [first, second])
        if with_strip_colon_and_index:
            names = map(name_str_without_colon_and_index, names)
        names = list(names)
        self.assertEqual(names[0], names[1], 'Name not equal.')