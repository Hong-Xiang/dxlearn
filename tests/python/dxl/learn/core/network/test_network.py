import unittest

from dxl.learn.core import Network
import pytest

@pytest.mark.skip(reason='not impl yet')
class TestNetwork(unittest.TestCase):
    def make_network_with_dummy_output_and_mse_objective(
            self, is_add_trainer=True):
        pass

    def make_dummy_dataset(self):
        pass

    def test_add_trainer(self):
        n = self.make_network_with_dummy_output_and_mse_objective()
        t = n.subgraph(n.KEYS.SUBGRAPH.TRAINER)
        self.assertIs(
            t.tensor(t.KEYS.TENSOR.OBJECTIVE), n.tensor(n.KEYS.TENSOR.LOSS))
        self.assertIs(
            n.tensor(n.KEYS.TENSOR.TRAIN), t.tensor(t.KEYS.TENSOR.MAIN))

    def test_no_trainer(self):
        n = self.make_network_with_dummy_output_and_mse_objective(
            is_add_trainer=False)
        self.assertIsNone(n.subgraph(n.SUBGRAPH.TRAINER))
        self.assertIsNone(n.tensor(n.TENSOR.LOSS))

    def test_saver(self):
        pass

    def test_loader(self):
        pass