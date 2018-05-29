import unittest
import dxl.learn.utils.general as dlug


class TestStripColonAndIndexFromName(unittest.TestCase):
    def test_with_colon_and_index(self):
        assert dlug.strip_colon_and_index_from_name('a:0') == 'a'

    def test_without_colon_and_index(self):
        assert dlug.strip_colon_and_index_from_name('a') == 'a'