from pathlib import Path
import os


def sandbox():
    from dxl.learn.backend import current_backend
    return current_backend().sandbox


def test_resource_root():
    os.getenv('PATH_DXLEARN_TEST_RESOURCE')


def name_str_without_colon_and_index(name):
    from .general import strip_colon_and_index_from_name
    return strip_colon_and_index_from_name(str(name))


def get_object_name_str(obj_with_name):
    if hasattr(obj_with_name, 'info'):
        return str(obj_with_name.info.name)
    if hasattr(obj_with_name, 'name'):
        return str(obj_with_name.name)
    if isinstance(obj_with_name, (str, Path)):
        return str(obj_with_name)
    raise TypeError("Invalid obj_with_name {}.".format(obj_with_name))
