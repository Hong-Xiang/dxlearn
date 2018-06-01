import unittest
from dxl.learn.distribute import *

import pytest
import tensorflow as tf

from dxl.learn.test import DistributeTestCase


class ClusterTestCase(DistributeTestCase):
    def get_ps_worker_config(self):
        return {
            JOB_NAME.WORKER: [
                "worker0.example.com:2222", "worker1.example.com:2222",
                "worker2.example.com:2222"
            ],
            JOB_NAME.PARAMETER_SERVER:
            ["ps0.example.com:2222", "ps1.example.com:2222"]
        }

    def get_ps_worker_jobs(self):
        return {JOB_NAME.WORKER, JOB_NAME.PARAMETER_SERVER}

    def get_master_worker_config(self):
        return {
            JOB_NAME.MASTER: ['localhost:2221'],
            JOB_NAME.WORKER: ['localhost:2333', 'localhost:2334']
        }

    def get_master_worker_jobs(self):
        return {JOB_NAME.MASTER, JOB_NAME.WORKER}

    def get_master_host(self):
        return Host(JOB_NAME.MASTER, 0, 'localhost', 2221)

    def get_worker_host(self, task_index):
        return Host(JOB_NAME.WORKER, task_index, 'localhost',
                    2333 + task_index)


class TestClusterSpec(ClusterTestCase):
    def test_jobs(self):
        c = ClusterSpec(self.get_ps_worker_config())
        self.assertEqual(set(self.get_ps_worker_jobs()), set(c.jobs))


class TestMasterWokerSpec(ClusterTestCase):
    def test_jobs(self):
        c = MasterWorkerClusterSpec(self.get_master_worker_config())
        self.assertEqual(set(self.get_master_worker_jobs()), set(c.jobs))

    def test_nb_workers(self):
        c = MasterWorkerClusterSpec(self.get_master_worker_config())
        assert c.nb_workers == len(
            self.get_master_worker_config()[JOB_NAME.WORKER])

    def test_master(self):
        c = MasterWorkerClusterSpec(self.get_master_worker_config())
        assert c.master == self.get_master_worker_config()[JOB_NAME.MASTER]

    def test_worker(self):
        c = MasterWorkerClusterSpec(self.get_master_worker_config())
        assert c.worker == self.get_master_worker_config()[JOB_NAME.WORKER]


class TestCluster(ClusterTestCase):
    def test_hosts_master(self):
        spec = ClusterSpec(self.get_master_worker_config())
        cluster = Cluster(spec)
        assert cluster.hosts[JOB_NAME.MASTER][0] == Host(
            JOB_NAME.MASTER, 0, 'localhost', 2221)

    def test_hosts_worker(self):
        spec = ClusterSpec(self.get_master_worker_config())
        cluster = Cluster(spec)
        for i in range(2):
            assert cluster.hosts[JOB_NAME.WORKER][i] == Host(
                JOB_NAME.WORKER, i, 'localhost', 2333 + i)


class TestMasterWorkerCluster(ClusterTestCase):
    def test_master(self):
        spec = ClusterSpec(self.get_master_worker_config())
        cluster = MasterWorkerCluster(spec)
        assert cluster.master() == self.get_master_host()

    def test_worker(self):
        spec = ClusterSpec(self.get_master_worker_config())
        cluster = MasterWorkerCluster(spec)
        for i in range(2):
            assert cluster.worker(i) == self.get_worker_host(i)


class TestDefaultCluster(ClusterTestCase):
    def get_a_cluster(self):
        return MasterWorkerCluster(
            MasterWorkerClusterSpec(self.get_master_worker_config()))

    def test_cluster(self):
        c = self.get_a_cluster()
        DefaultCluster.set(c)
        self.assertIsInstance(DefaultCluster.cluster(), MasterWorkerCluster)
        self.assertEqual(self.get_master_host(), DefaultCluster.cluster().master())


# class TestMakeClusterWithMaster(ClusterTestCase):
#     def test_cluster_created(self):
#         make_master_worker_cluster(
#             MasterWorkerClusterSpec(self.get_master_worker_cluster_config()),
#             JOB_NAME.WORKER, 1)
#         assert ThisHost.host().task_index == 1
#         assert ThisHost.host().job == JOB_NAME.WORKER
