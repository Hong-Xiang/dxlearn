import unittest
import dxl.learn.core.config as dlcc


class TestConfigurableWithName(unittest.TestCase):
    def setUp(self):
        dlcc.DefaultConfig.reset()

    def tearDown(self):
        dlcc.DefaultConfig.reset()

    def test_basic(self):
        c = dlcc.ConfigurableWithName('x', {'a': 1, 'b': 2})
        self.assertEqual(c.config('a'), 1)

    def test_baisc2(self):
        dlcc.DefaultConfig.reset()
        c = dlcc.ConfigurableWithName('x/y', {'a': 1, 'b': 2})
        self.assertEqual(c.config('a'), 1)

    def test_inherent(self):
        dlcc.DefaultConfig.reset()
        c0 = dlcc.ConfigurableWithName('x', {'a': 0})
        self.assertEqual(c0.config('a'), 0)
        c1 = dlcc.ConfigurableWithName('x/y')
        self.assertEqual(c1.config('a'), 0)

    def test_default(self):
        dlcc.DefaultConfig.reset()

        class A(dlcc.ConfigurableWithName):
            @classmethod
            def default_config(cls):
                return {'a': 1}

        c1 = A('x')
        self.assertEqual(c1.config('a'), 1)
        c2 = A('x', {'a': None})
        self.assertEqual(c2.config('a'), 1)
        c3 = A('x', {'a': 2})
        self.assertEqual(c3.config('a'), 2)

    def test_inherent_default(self):
        dlcc.DefaultConfig.reset()

        class A(dlcc.ConfigurableWithName):
            @classmethod
            def default_config(cls):
                return {'a': 1}

        c0 = dlcc.ConfigurableWithName('x', {'a': 0})
        self.assertEqual(c0.config('a'), 0)
        c1 = A('x/y')
        self.assertEqual(c1.config('a'), 0)
        c2 = A('x/y', {'a': None})
        self.assertEqual(c2.config('a'), 0)
        c3 = A('x/y', {'a': 2})
        self.assertEqual(c3.config('a'), 2)
        c4 = A('z')
        self.assertEqual(c4.config('a'), 1)

    def test_external_config(self):
        dlcc.DefaultConfig.update('x', {'key': 'value'})
        c = dlcc.ConfigurableWithName('x')
        self.assertEqual(c.config('x')['key'], 'value')

    def test_update_config(self):
        dlcc.update_config('name', {'key': 'value'})
        assert dlcc.ConfigurableWithName('name').config('key') == 'value'

    def test_update_config2(self):
        dlcc.update_config('name/sub', {'key': 'value'})
        assert dlcc.ConfigurableWithName('name/sub').config('key') == 'value'

    def test_update_config_inherence(self):
        dlcc.update_config('name', {'key': 'value'})
        assert dlcc.ConfigurableWithName('name/sub').config('key') == 'value'
