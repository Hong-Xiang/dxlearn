from dxl.learn.test import DistributeTestCase
from dxl.learn.distribute.host import Host, Master, ThisHost
import pytest


class TestHost(DistributeTestCase):
    def test_eq(self):
        h1 = Host('master')
        h2 = Host('master')
        self.assertEqual(h1, h2)

    def test_eq(self):
        h1 = Host('worker')
        h2 = Host('worker', 0)
        self.assertEqual(h1, h2)

    def test_neq(self):
        h1 = Host('worker', 1)
        h2 = Host('worker', 0)
        self.assertNotEqual(h1, h2)

    def test_eq_with_ip(self):
        h1 = Host('w', 0, 'host0')
        h2 = Host('w', 0)
        self.assertEqual(h1, h2)

    def test_neq_with_ip(self):
        h1 = Host('w', 0, 'host0')
        h2 = Host('w', 0, 'host1')
        self.assertNotEqual(h1, h2)

    def test_device_prefix(self):
        h = Host('master', 3)
        self.assertEqual(h.device_prefix(), '/job:master/task:3')


class TestMaster(DistributeTestCase):
    def setUp(self):
        super().setUp()
        Master.reset()

    def tearDown(self):
        Master.reset()
        super().tearDown()

    def test_set(self):
        h = Host('master')
        Master.set(h)
        self.assertEqual(Master.host(), h)

    def test_is_master(self):
        h = Host('master')
        Master.set(h)
        self.assertTrue(Master.is_master(h))
        self.assertTrue(Master.is_master(Host('master', 0, 'host0')))
        self.assertTrue(not Master.is_master(Host('worker', 0)))


class TestThisHost(DistributeTestCase):
    def setUp(self):
        super().setUp()
        ThisHost.reset()

    def tearDown(self):
        ThisHost.reset()
        super().tearDown()

    def get_host(self):
        return Host('worker', 1)

    def get_another_host(self):
        return Host('master', 0)

    def test_set(self):
        h = self.get_host()
        ThisHost.set(h)
        self.assertEqual(ThisHost.host(), h)

    def test_is_this(self):
        ThisHost.set(self.get_host())
        self.assertTrue(ThisHost.is_this(self.get_host()))

    def test_not_this(self):
        ThisHost.set(self.get_host())
        self.assertTrue(not ThisHost.is_this(self.get_another_host()))

    def test_is_master(self):
        Master.set(self.get_host())
        ThisHost.set(self.get_host())
        self.assertTrue(ThisHost.is_master())
        Master.reset()
