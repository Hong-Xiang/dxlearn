import unittest
import dxl.learn.utils.general as dlug
import pytest


@pytest.mark.parametrize("test_input,expected", [('a:0', 'a'),
                                                 ('scope/x:0', 'scope/x')])
def test_with_colon_and_index(test_input, expected):
    assert dlug.strip_colon_and_index_from_name(test_input) == expected


def test_without_colon_and_index():
    assert dlug.strip_colon_and_index_from_name('a') == 'a'
