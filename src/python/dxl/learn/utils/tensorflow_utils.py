import tensorflow as tf

from typing import NamedTuple, Dict


class SimplifiedEvent(NamedTuple):
    step: int
    values: Dict[str, float] = {}


def parse_event(e):
    step = e.step
    values = {v.tag: v.simple_value for v in e.summary.value}
    return SimplifiedEvent(step, values)


def load_tensorboard(path):
    return [parse_event(e) for e in tf.train.summary_iterator(path)]


def is_with_fields(fields, event):
    if fields is None:
        return True
    for k in fields:
        if not k in event.values:
            return False
    return True


class TensorboardEvents:
    """
    A List of Events, each event is a collection of {tag, simple_value}
    """

    def __init__(self, path, fields=None, tag2field_mapping=None):
        self.path = path
        self.fields = fields
        self.tag2field_mapping = tag2field_mapping or {}
        self._cache = []
        self._loaded = False

    def parse_event(self, e):
        result = {}
        for v in e.summary.value:
            tag = self.tag2field_mapping.get(v.tag, v.tag)
            result[tag] = v.simple_value
        return SimplifiedEvent(e.step, result)

    def tensorboard_iterator(self):
        return tf.train.summary_iterator(self.path)

    def __iter__(self):
        if self._loaded:
            return iter(self._cache)
        return (self.parse_event(e) for e in self.tensorboard_iterator()
                if is_with_fields(self.fields, self.parse_event(e)))

    def load_all(self):
        self._cache = list(iter(self))
        self._loaded = True

    def __getitem__(self, k):
        if not self._loaded:
            self.load_all()
        if isinstance(k, int):
            return self._cache[k]
        if isinstance(k, str):
            return [d.values[k] for d in self._cache]
        raise TypeError(f"Unknown key type {type(k)}")
