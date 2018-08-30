import pytest
import tensorflow as tf


@pytest.fixture(scope="module")
def tensorflow_test():
    with tf.Graph().as_default():
        yield


def pytest_collection_modifyitems(session, config, items):
    items[:] = [item for item in items if item.name != 'test_session']
