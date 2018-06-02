import os
from pathlib import Path
import numpy as np
import uuid
from dxl.learn.backend import current_backend
from dxl.learn.utils.test_utils import name_str_without_colon_and_index, get_object_name_str
from contextlib import contextmanager
import tensorflow as tf
import unittest
import pytest
from unittest.mock import patch


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

    def make_dummy_variable(self, info=None):
        from dxl.learn.core import Variable
        if info is None:
            info == str(uuid.uuid4())
        return Variable(info, [])

    @property
    def resource_path(self):
        return Path(os.getenv('DEV_DXLEARN_TEST_RESOURCE_PATH'))

    def assertFloatArrayEqual(self, first, second, msg=None):
        return np.testing.assert_array_almost_equal(
            np.array(first), np.array(second), err_msg=msg)

    def assertNameEqual(self, first, second, with_strip_colon_and_index=True):
        names = map(get_object_name_str, [first, second])
        if with_strip_colon_and_index:
            names = map(name_str_without_colon_and_index, names)
        names = list(names)
        self.assertEqual(names[0], names[1], 'Name not equal.')

    @pytest.mark.skip
    # @unittest.skip
    @contextmanager
    def test_session(self):
        with super().test_session() as sess:
            yield sess

    @contextmanager
    def variables_initialized_test_session(self):
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            yield sess


class DistributeTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.train_server_patch = patch('tensorflow.train.Server')
        m = self.train_server_patch.start()

    def tearDown(self):
        from dxl.learn.distribute import reset_cluster
        reset_cluster()
        self.train_server_patch.stop()
        super().tearDown()
