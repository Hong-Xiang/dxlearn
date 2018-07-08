from typing import Dict, List, TypeVar
import arrow
import sys
import tensorflow as tf

from dxpy.collections.dicts import combine_dicts
from dxpy.configs import configurable
from dxpy.core.path import Path
from dxpy.learn.config import config
from dxpy.learn.session import get_default_session
import arrow
import time
from ..graph import Graph, NodeKeys
import click


def get_summary_type(tensor, summary_type):
    from dxpy.learn.model.tensor import shape_as_list
    if summary_type is None:
        shape = shape_as_list(tensor)
        if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1):
            summary_type = 'scalar'
        elif len(shape) == 4:
            summary_type = 'image'
        else:
            raise TypeError(
                "Can not auto infer summary type of {}.".format(tensor))
    return summary_type


def get_merge_method(tensor, merge_method):
    if merge_method is None:
        if get_summary_type(tensor, None) == 'scalar':
            return 'mean'
        elif get_summary_type(tensor, None) == 'image':
            return 'concat'
    else:
        return merge_method


def create_summary_op(name, tensor, summary_type=None):
    summary_type = get_summary_type(tensor, summary_type)
    if summary_type == 'scalar':
        return tf.summary.scalar(name, tensor)
    elif summary_type == 'image':
        return tf.summary.image(name, tensor)
    raise ValueError("Unknown summary_type {}.".format(summary_type))


class SummaryKeys:
    MERGED = 'merged'


class SummaryWriter(Graph):
    """
    Summary writer class.
    Usage:

        #create dataset, network
        dataset = ...
        network = ...
        summary = SummaryWriter('train', {'loss': network['loss']}, {'input': network['input'], 'label': network['label']}, path='./summary/train/)

        #After create session
        sess = tf.Session()
        with sess.as_default():
            dataset.post_session_created()
            network.post_session_created()
            summary.post_session_created()

        #Train steps:
        with sess.as_default():
            for i in range(1000):
                network.train()
                if (i % 10) == 0:
                    summary.summary({'input': input_ndarray, 'label': label_ndarray})
            summary.flust()
    """
    @configurable(config, with_name=True)
    def __init__(self, name: str='summary',
                 tensors: Dict[str, tf.Tensor]=None,
                 additional_inputs: Dict[str, tf.Tensor]=None,
                 nb_runs: Dict[str, int]=None,
                 merge_method: Dict[str, str]=None,
                 *,
                 path: str='./summary/train',
                 nb_max_image: int=3,
                 interval: int=600,
                 with_prefix: bool=False,
                 **kw):
        """
        Args:
            tensors: dict of name: SummaryItem
            inputs: tensors required to run summary (will used for feeds)
        """
        super().__init__(name, path=path,
                         nb_max_image=nb_max_image, with_prefix=with_prefix,
                         interval=interval, **kw)
        if tensors is None:
            tensors = dict()
        if additional_inputs is None:
            additional_inputs = dict()
        if nb_runs is None:
            nb_runs = dict()
        if merge_method is None:
            merge_method = dict()
        self.nb_runs = nb_runs
        self.merge_method = merge_method
        self.multi_runs = dict()
        self.nb_max_runs = 0
        self.inputs = dict()
        with tf.name_scope(self.basename):
            tensors = self._processing(tensors)
            summary_ops = self._create_summary_ops(tensors)
            with tf.name_scope('merge_summary_ops'):
                self.register_node(SummaryKeys.MERGED,
                                   tf.summary.merge(summary_ops))
        self.register_task('create_writer', self._create_writer)
        self.register_task('flush', self.flush)
        self._pre_summ = arrow.now()

    def _processing(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        result = dict()
        for k in tensors:
            if k in self.nb_runs:
                self.inputs[k] = tensors[k]
                with tf.name_scope('multi_runs'):
                    self.multi_runs[k] = [tf.placeholder(
                        tensors[k].dtype, tensors[k].shape, name=k) for i in range(self.nb_runs[k])]
                    if self.nb_runs[k] > self.nb_max_runs:
                        self.nb_max_runs = self.nb_runs[k]
                    mm = get_merge_method(tensors[k], self.merge_method.get(k))
                    if mm == 'mean':
                        reduce_mean = tf.add_n(
                            self.multi_runs[k]) / len(self.multi_runs[k])
                        result[k] = reduce_mean
                    elif mm == 'concat':
                        concat = tf.concat(self.multi_runs[k], axis=0)
                        result[k] = concat
            else:
                result[k] = tensors[k]
        return result

    def _create_summary_ops(self, tensors):
        result = list()
        for k in tensors:
            summ_op = create_summary_op(k, tensors[k])
            result.append(summ_op)
        return result

    def post_session_created(self):
        self.run('create_writer')

    def _get_multiple_run_feeds(self, feeds=None):
        result = dict()
        for i in range(self.nb_max_runs):
            current_result = get_default_session().run(
                self.inputs, self.get_feed_dict(feeds))
            for k in self.inputs:
                if k in self.nb_runs and self.nb_runs[k] > i:
                    result[self.multi_runs[k][i]] = current_result[k]
        return result

    def summary(self, feeds=None):
        from ..scalar import current_step
        mrf = self._get_multiple_run_feeds(feeds)
        feed_dict = self.get_feed_dict(feeds)
        feed_dict.update(mrf)
        value = get_default_session().run(
            self.nodes[SummaryKeys.MERGED], feed_dict=feed_dict)
        self.nodes['summary_writer'].add_summary(value, current_step())

    def auto_summary(self, feeds=None):
        time_now = arrow.now()
        dtime = (time_now - self._pre_summ).seconds
        if dtime > self.param('interval'):
            self.summary(feeds)
            print('Add Summary at {}'.format(arrow.now()))
            sys.stdout.flush()
            self._pre_summ = time_now
        else:
            time.sleep(self.param('interval') - dtime)

    def flush(self):
        self.nodes['summary_writer'].flush()

    def _create_writer(self, feeds):
        self.register_node('summary_writer',
                           tf.summary.FileWriter(self.param('path', feeds),
                                                 get_default_session().graph))


class SummaryMaker(Graph):
    """
    """
    @configurable(config, with_name=True)
    def __init__(self,
                 name: str='summary',
                 tensors: Dict[str, tf.Tensor]=None,
                 *,
                 nb_max_image: int = 3,
                 **kw):
        """
        Args:
            tensors: dict of name: SummaryItem
            inputs: tensors required to run summary (will used for feeds)
        """
        super().__init__(name, nb_max_image=nb_max_image, **kw)
        if tensors is None:
            tensors = dict()
        with tf.name_scope(self.basename):
            tensors = self._processing(tensors)
            summary_ops = self._create_summary_ops(tensors)
        self.register_task('create_writer', self._create_writer)
        self.register_task('flush', self.flush)
        self._pre_summ = arrow.now()

    def _processing(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return tensors

    def _create_summary_ops(self, tensors):
        result = list()
        for k in tensors:
            summ_op = create_summary_op(k, tensors[k])
            result.append(summ_op)
        return result

    def post_session_created(self):
        self.run('create_writer')

    def _get_multiple_run_feeds(self, feeds=None):
        result = dict()
        for i in range(self.nb_max_runs):
            current_result = get_default_session().run(
                self.inputs, self.get_feed_dict(feeds))
            for k in self.inputs:
                if k in self.nb_runs and self.nb_runs[k] > i:
                    result[self.multi_runs[k][i]] = current_result[k]
        return result

    def summary(self, feeds=None):
        from ..scalar import current_step
        mrf = self._get_multiple_run_feeds(feeds)
        feed_dict = self.get_feed_dict(feeds)
        feed_dict.update(mrf)
        value = get_default_session().run(
            self.nodes[SummaryKeys.MERGED], feed_dict=feed_dict)
        self.nodes['summary_writer'].add_summary(value, current_step())

    def auto_summary(self, feeds=None):
        time_now = arrow.now()
        dtime = (time_now - self._pre_summ).seconds
        if dtime > self.param('interval'):
            self.summary(feeds)
            print('Add Summary at {}'.format(arrow.now()))
            sys.stdout.flush()
            self._pre_summ = time_now
        else:
            time.sleep(self.param('interval') - dtime)

    def flush(self):
        self.nodes['summary_writer'].flush()

    def _create_writer(self, feeds):
        self.register_node('summary_writer',
                           tf.summary.FileWriter(self.param('path', feeds),
                                                 get_default_session().graph))
