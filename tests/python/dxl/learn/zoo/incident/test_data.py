from dxl.learn.test import TestCase
from dxl.data.zoo.incident import load_table, Hit, Photon, Coincidence
from dxl.learn.zoo.incident.data import reindex_crystal

# from dxl.learn.zoo.incident.data import create_dataset, dataset_db

import os
path_p5 = os.environ['GHOME'] + \
    '/Workspace/IncidentEstimation/data/gamma_photo_5.h5'


def test_reindex_crystal():
    p = Photon([
        Hit(0.0, 0.0, 0.0, 0.0, crystal_index=12),
        Hit(0.0, 0.0, 0.0, 0.0, crystal_index=23),
        Hit(0.0, 0.0, 0.0, 0.0, crystal_index=12),
    ])
    p2 = reindex_crystal(p)
    assert p2.hits[0].crystal_index == 0
    assert p2.hits[1].crystal_index == 1
    assert p2.hits[0].crystal_index == 0
