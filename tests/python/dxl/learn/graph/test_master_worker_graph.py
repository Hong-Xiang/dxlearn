from dxl.learn.graph import MasterWorkerTaskBase
import tensorflow as tf
import pytest


class DoNothing:
    def __init__(self, *args, **kwargs):
        pass


def test_minium(monkeypatch):
    monkeypatch.setattr(tf.train, 'Server', DoNothing)
    mwt = MasterWorkerTaskBase(job=MasterWorkerTaskBase.KEYS.SUBGRAPH.MASTER)
    assert mwt.job == 'master'
    assert mwt.task_index == 0
    assert mwt.nb_workers == 2