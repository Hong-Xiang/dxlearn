import unittest
from dxl.learn.core.distribute import Host, Cluster, ClusterSpec
from dxl.learn.core import distribute as dlcd


class TestHost(unittest.TestCase):
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


class TestClusterSpec(unittest.TestCase):
    def test_basic(self):
        cspec = ClusterSpec()


class TestCluster(unittest.TestCase):
    def test_cluster(self):
        cfg = {
            "worker": [
                "worker0.example.com:2222", "worker1.example.com:2222",
                "worker2.example.com:2222"
            ],
            "ps": ["ps0.example.com:2222", "ps1.example.com:2222"]
        }
        Cluster.set(ClusterSpec(cfg))
        self.assertIn(Host('ps', 0, 'ps0.example.com', 2222), Cluster.hosts())
        Cluster.reset()


class TestHelperFunction(unittest.TestCase):
    def test_make_cluster(self):
        dlcd.make_distribute_host(dlcd.ClusterSpec(), dlcd.JOB_NAME.WORKER, 1)
        self.assertEqual(dlcd.ThisHost.host().task_index, 1)
        self.assertEqual(dlcd.ThisHost.host().job, dlcd.JOB_NAME.WORKER)
        dlcd.Cluster.reset()
        dlcd.Server.reset()
        dlcd.ThisHost.reset()
        dlcd.MasterHost.reset()