import pytest
from dxl.learn.backend import current_backend


@pytest.fixture()
def sandbox():
    with current_backend().sandbox():
        yield
