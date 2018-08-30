import pytest
from dxl.learn.backend import current_backend
import tensorflow as tf


@pytest.fixture()
def sandbox():
    with current_backend().sandbox():
        yield


@pytest.fixture()
def tensorflow_sandbox():
    with tf.Graph().as_default():
        yield
