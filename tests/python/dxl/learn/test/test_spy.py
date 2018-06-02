from dxl.learn.test import TestCase
from dxl.learn.core import Tensor

from dxl.learn.test import tensor_run_spy


class TestRunSpy(TestCase):
    def test_basic(self):
        o, v = tensor_run_spy()
        assert isinstance(o, Tensor)
        with self.variables_initialized_test_session() as sess:
            assert sess.run(v.data) == 0
            for i in range(3):
                sess.run(o.data)
                assert sess.run(v.data) == i + 1
