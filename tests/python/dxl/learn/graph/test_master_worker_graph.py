from dxl.learn.graph import MasterWorkerTaskBase
import tensorflow as tf
import pytest

from dxl.learn.test import DistributeTestCase
from dxl.learn.core import Host


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

    def assertCorrectlyInitialized(self, graph, job, task_index, nb_workers):
        assert graph.job == job
        assert graph.task_index == task_index
        assert graph.nb_workers == nb_workers
        assert graph.is_cluster_init == True
        assert graph.hosts['master'].ip == 'host0'
        assert graph.hosts['master'].port == 2222
        for i in range(nb_workers):
            assert graph.hosts['worker'][i].ip == 'host1'
            assert graph.hosts['worker'][i].port == 2222 + i

    def test_construct(self):
        job = 'master'
        nb_workers = 3
        mwt = self.get_graph(job, nb_workers=nb_workers)
        self.assertCorrectlyInitialized(mwt, job, 0, nb_workers)

    def test_construct_worker(self):
        job = 'worker'
        task_index = 1
        nb_workers = 3
        mwt = self.get_graph(job, task_index, nb_workers)
        self.assertCorrectlyInitialized(mwt, job, task_index, nb_workers)

    def test_master_host(self):
        from dxl.learn.core import MasterHost
        g = self.get_graph()
        assert g.master_host() is MasterHost.host()