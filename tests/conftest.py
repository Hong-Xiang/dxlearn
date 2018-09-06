import pytest
import tensorflow as tf
from dxl.learn.config import clear_config


@pytest.fixture(scope="module")
def tensorflow_test():
    with tf.Graph().as_default():
        yield


@pytest.fixture(scope="module")
def tensorflow_test_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        yield sess


@pytest.fixture(scope="function")
def clean_config():
    clear_config()
    yield
    clear_config()


def pytest_collection_modifyitems(session, config, items):
    items[:] = [item for item in items if item.name != 'test_session']
