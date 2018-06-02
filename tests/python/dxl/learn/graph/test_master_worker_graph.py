import unittest

import pytest
import tensorflow as tf

from dxl.learn.distribute import (Host, Master, make_master_worker_cluster,
                                  DistributeGraphInfo)
from dxl.learn.graph import MasterWorkerTaskBase
from dxl.learn.test import DistributeTestCase


class TestMasterWorkerTaskBase(DistributeTestCase):
    def get_cluster(self, job, task_index, nb_workers=3):
        return make_master_worker_cluster(
            self.get_cluster_config(nb_workers), job, task_index)

    def test_get_cluster_master(self):
        c = self.get_cluster('master', 0, 3)
        assert c.nb_workers == 3

    def get_graph(self, job='master', task_index=0, nb_workers=3):
        return MasterWorkerTaskBase(
            job=job,
            task_index=task_index,
            cluster=self.get_cluster(job, task_index, nb_workers))

    def get_cluster_config(self, nb_workers):
        return {
            'master': ['host0:2222'],
            'worker': ['host1:{}'.format(2222 + i) for i in range(nb_workers)]
        }

    def assertCorrectlyInitialized(self, graph, job, task_index, nb_workers):
        assert graph.job == job
        assert graph.task_index == task_index
        assert graph.nb_workers == nb_workers
        assert graph._cluster is not None
        assert graph.master().ip == 'host0'
        assert graph.master().port == 2222
        for i in range(nb_workers):
            assert graph.worker(i).ip == 'host1'
            assert graph.worker(i).port == 2222 + i

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
        g = self.get_graph()
        assert g.master() == Master.host()
        assert g.master().ip == Master.host().ip

    def test_worker(self):
        g = self.get_graph()
        assert g.worker(0) == Host('worker', 0)
        assert g.worker(1) == Host('worker', 1)

    def get_function_to_check_called_with_decorator(self, deco):
        class NotExpectedToBeCalled(Exception):
            pass

        @deco
        def foo():
            raise NotExpectedToBeCalled()

        return foo, NotExpectedToBeCalled

    def test_master_only_on_master(self):
        g = self.get_graph('master')
        foo, e = self.get_function_to_check_called_with_decorator(
            g.master_only)
        with pytest.raises(e):
            foo()

    def test_master_only_on_worker(self):
        g = self.get_graph('worker')
        foo, e = self.get_function_to_check_called_with_decorator(
            g.master_only)
        foo()

    def test_worker_only_on_master(self):
        g = self.get_graph('master')
        foo, e = self.get_function_to_check_called_with_decorator(
            g.worker_only)
        foo()

    def test_worker_only_on_worker(self):
        g = self.get_graph('worker')
        foo, e = self.get_function_to_check_called_with_decorator(
            g.worker_only)
        with pytest.raises(e):
            foo()

    def test_graph_info(self):
        g = self.get_graph('master')
        self.assertIsInstance(g.info, DistributeGraphInfo)
        assert g.info.host.job == 'master'
