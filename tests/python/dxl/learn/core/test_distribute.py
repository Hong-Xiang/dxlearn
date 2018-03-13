import unittest
from dxl.learn.core.distribute import Host, Cluster


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


class TestCluster(unittest.TestCase):
    def test_cluster(self):
        cfg = {"worker": ["worker0.example.com:2222",
                          "worker1.example.com:2222",
                          "worker2.example.com:2222"],
               "ps": ["ps0.example.com:2222",
                      "ps1.example.com:2222"]}
        Cluster.set_cluster(cfg)
        self.assertIn(Host('ps', 0, 'ps0.example.com', 2222), Cluster.hosts())
        

