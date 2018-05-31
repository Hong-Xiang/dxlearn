from dxl.learn.graph import MasterWorkerTaskBase
import tensorflow as tf
import pytest

from dxl.learn.test import DistributeTestCase


class DoNothing:
    def __init__(self, *args, **kwargs):
        pass


class TestMasterWorkerTaskBase(DistributeTestCase):
    def get_graph(self, job='master', task_index=0, nb_workers=3):
        return MasterWorkerTaskBase(
            job=job,
            task_index=task_index,
            cluster_config=self.get_cluster_config(nb_workers))

    def get_cluster_config(self, nb_workers):
        return {
            'master': ['host0:2222'],
            'worker': ['host1:{}'.format(2222 + i) for i in range(nb_workers)]
        }

    def test_construct(self):
        job = 'master'
        nb_workers = 3
        mwt = self.get_graph(job, nb_workers=nb_workers)
        assert mwt.job == job
        assert mwt.task_index == 0
        assert mwt.nb_workers == nb_workers

    def test_construct_worker(self):
        job = 'worker'
        task_index = 1
        nb_workers = 3
        mwt = self.get_graph(job, task_index, nb_workers)
        assert mwt.job == job
        assert mwt.task_index == task_index
        assert mwt.nb_workers == nb_workers
