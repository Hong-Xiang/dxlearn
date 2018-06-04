import re


def device_name(device_type, device_id=None):
    if device_id is None:
        device_id = 0
    return '/{t}:{i}'.format(t=device_type, i=device_id)


_pattern_colon_and_index = re.compile('\A([\w/\-_]+)(:\d+)?')


def strip_colon_and_index_from_name(s):
    s = str(s)
    m = _pattern_colon_and_index.match(s)
    if m:
        return str(m[1])
    else:
        raise ValueError("Unsupported name string: {}.".format(s))


def refined_tensor_or_graph_name(tensor_or_graph):
    import tensorflow as tf
    if isinstance(tensor_or_graph, tf.Tensor):
        return tensor_or_graph.name.replace(':', '_')
    return tensor_or_graph.name


def pre_work(device=None):
    import tensorflow as tf
    from dxpy.learn.scalar import create_global_scalars
    if device is None:
        create_global_scalars()
    else:
        with tf.device(device):
            create_global_scalars()


def load_yaml_config(filename='dxln.yml'):
    from ..config import config
    import yaml
    with open(filename) as fin:
        cfg = yaml.load(fin)
    config.update(cfg)
    if 'include' in cfg:
        for f in cfg['include']:
            with open(f) as fin:
                cfg_i = yaml.load(fin)
                config.update(cfg_i)


def unbox(obj):
    """
    General unbox helper.
    """
    if hasattr(obj, 'unbox'):
        return obj.unbox()
    return obj


def generic_map(func, collection):
    if isinstance(collection, list):
        return [func(v) for v in collection]
    if isinstance(collection, tuple):
        return tuple([func(v) for v in collection])
    if isinstance(collection, dict):
        return {k: func(v) for k, v in collection.items()}
    if isinstance(collection, set):
        return {func(v) for v in collection}
    raise TypeError("Unspported collection type: {}.".format(type(collection)))
