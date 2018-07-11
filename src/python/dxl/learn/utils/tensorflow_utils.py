import tensorflow as tf

from typing import NamedTuple, Dict


class SimplifiedSummary(NamedTuple):
    step: int
    values: Dict[str, float]


def parse_event(e):
    step = e.step
    values = {v.tag: v.simple_value for v in e.summary.value}
    return SimplifiedSummary(step, values)


def load_tensorboard(path):
    return [parse_event(e) for e in tf.train.summary_iterator(path)]
