import unittest
from dxl.learn.core.distribute import Host, Cluster, ClusterSpec
from dxl.learn.core import distribute as dlcd
import pytest
import tensorflow as tf

from dxl.learn.test import DistributeTestCase


class TestHost(DistributeTestCase):
    def test_eq(self):
        h1 = Host('master')
        h2 = Host('master')
        self.assertEqual(h1, h2)
        h1 = Host('worker', 0)
        h2 = Host('worker', 0)
        self.assertEqual(h1, h2)
        h1 = Host('worker', 1)
        h2 = Host('worker', 0)
        self.assertNotEqual(h1, h2)


class TestClusterSpec(DistributeTestCase):
    def test_basic(self):
        cspec = ClusterSpec(dlcd.DEFAULT_CLUSTER_CONFIG)


class TestCluster(DistributeTestCase):
    def get_config(self):
        return {
            "worker": [
                "worker0.example.com:2222", "worker1.example.com:2222",
                "worker2.example.com:2222"
            ],
            "ps": ["ps0.example.com:2222", "ps1.example.com:2222"]
        }

    def test_cluster(self):
        Cluster.set(ClusterSpec(self.get_config()))
        self.assertIn(Host('ps', 0, 'ps0.example.com', 2222), Cluster.hosts())
        Cluster.reset()


class DoNothing:
    def __init__(self, *args, **kwargs):
        pass


class TestMakeClusterWithMaster(DistributeTestCase):
    def get_config(self, nb_workers):
        return {
            'master': ['host0:2222'],
            'worker': ['host1:{}'.format(2222 + i) for i in range(nb_workers)]
        }

    def test_cluster_created(self):
        dlcd.make_distribute_host

        dlcd.make_distribute_host(
            dlcd.ClusterSpec(dlcd.DEFAULT_CLUSTER_CONFIG),
            dlcd.JOB_NAME.WORKER, 1)
        assert dlcd.ThisHost.host().task_index == 1
        assert dlcd.ThisHost.host().job == dlcd.JOB_NAME.WORKER
