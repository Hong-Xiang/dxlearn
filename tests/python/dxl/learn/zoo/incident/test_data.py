from dxl.learn.test import TestCase

from dxl.learn.zoo.incident.data import create_dataset, dataset_db

import os
path_db = os.environ['GHOME'] + '/Workspace/IncidentEstimation/data/gamma.db'

import pytest


class TestDataset(TestCase):
    @pytest.mark.skip("slow")
    def test_load(self):
        dataset = create_dataset(dataset_db, path_db, 5, 32)
        samples = []
        with tf.Session() as sess:
            for i in range(1000):
                samples.append(sess.run(
                    [dataset.hits.data, dataset.first_hit_index.data, dataset.padded_size.data]))
        assert samples[0][0].shape == (32, 5, 4)
