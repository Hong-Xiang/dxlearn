from dxl.learn.test import TestCase
from dxl.learn.core import Tensor

from dxl.learn.test import OpRunSpy
import pytest


class TestRunSpy(TestCase):
    def test_nb_called(self):
        ors = OpRunSpy()
        with self.variables_initialized_test_session() as sess:
            assert sess.run(ors.nb_called.data) == 0
            for i in range(3):
                sess.run(ors.op.data)
                assert sess.run(ors.nb_called.data) == i + 1

    @pytest.mark.skip(reason='not correctly impl yet')
    def test_global_order(self):
        ors0 = OpRunSpy()
        with self.variables_initialized_test_session() as sess:
            assert sess.run(ors0.nb_called.data) == 0
            sess.run(ors0.op.data)
            assert sess.run(ors0.global_order.data) == 1
            sess.run(ors0.op.data)
            assert sess.run(ors0.global_order.data) == 2

    @pytest.mark.skip(reason='not correctly impl yet')
    def test_global_order_2(self):
        ors0 = OpRunSpy()
        ors1 = OpRunSpy()
        with self.variables_initialized_test_session() as sess:
            assert sess.run(ors0.nb_called.data) == 0
            assert sess.run(ors1.nb_called.data) == 0
            for i in range(3):
                sess.run(ors0.op.data)
                sess.run(ors1.op.data)
                assert sess.run(ors0.global_order.data) == i * 2 + 1
                assert sess.run(ors1.global_order.data) == i * 2 + 2

    @pytest.mark.skip(reason='not correctly impl yet')
    def test_skip_global_order(self):
        ors0 = OpRunSpy()
        ors1 = OpRunSpy(is_skip_global_order=True)
        ors2 = OpRunSpy()
        with self.variables_initialized_test_session() as sess:
            sess.run(ors0.op.data)
            sess.run(ors1.op.data)
            sess.run(ors2.op.data)
            assert sess.run(ors0.global_order.data) == 1
            assert ors1.global_order is None
            assert sess.run(ors2.global_order.data) == 2
