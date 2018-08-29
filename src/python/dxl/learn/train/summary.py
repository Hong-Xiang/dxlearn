import tensorflow as tf

from dxpy.collections.dicts import combine_dicts

from ..core import Graph
from ..graph import NodeKeys
import typing

from dxpy.learn.session import get_default_session


class SummaryItem:
    def __init__(self, tensor, stype=None):
        """
        Args:

        -   tensor: tensor to add to summary
        -   stype: `str`, indicating which summary type will be added.
                Currently support 'scalar', 'image'.
        """
        self.tensor = tensor
        if stype is None:
            if len(self.tensor.shape.as_list()) <= 1:
                stype = 'scalar'
            elif len(self.tensor.shape.as_list()) == 4:
                stype = 'image'
            else:
                raise TypeError(
                    "Can not auto infer summary type of {}.".format(tensor))
        self.stype = stype


class WithSummaryItems:
    def create_summary_items(self) -> typing.Dict[str, SummaryItem]:
        return dict()


class SummaryWriter(Graph):
    """
    Summary writer class.


    Usage:

        #create dataset, network

        dataset = ...

        network = ...

        summary = SummaryWriter('train', {'loss': network['loss']}, {'input': network['input'], 'label': network['label']} path='./summary/train/)

        #After create depsession
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

    def __init__(self,
                 name='summary',
                 tensors_to_summary=None,
                 inputs=None,
                 **config):
        """
        Inputs:
            tensors_to_summary: dict of name: SummaryItem
            inputs: tensors required to run summary (will used for feeds)
        """
        super().__init__(name, **config)
        if tensors_to_summary is None:
            tensors_to_summary = dict()
        if inputs is None:
            inputs = dict()

        self._tensors = combine_dicts(tensors_to_summary,
                                      self._default_summaries())
        for k in inputs:
            self.register_node(k, inputs[k])
        self._summary_ops = list()
        self.register_task('create_writer', self.__create_writer)
        self.register_task('flush', self.flush)
        self.register_main_task(self.summary)

        if self.param('add_prefix'):
            with tf.name_scope('summary'):
                self.__add_all_ops()
        else:
            self.__add_all_ops()

    @classmethod
    def _default_config(cls):
        return combine_dicts({
            'path': './summary',
            'add_prefix': False,
            'max_image': 3
        },
                             super()._default_config())

    @classmethod
    def _default_summaries(cls):
        return dict()

    def post_session_created(self):
        self.run('create_writer')

    def summary(self, feeds=None):
        from ..scalar import current_step
        if self.as_tensor() is None:
            return
        value = get_default_session().run(self.as_tensor(),
                                          self.get_feed_dict(feeds))
        self.nodes['summary_writer'].add_summary(value, current_step())

    def flush(self):
        self.nodes['summary_writer'].flush()

    def __add_all_ops(self):
        self.__add_tensors_to_summary()
        self.__add_main_op()

    def __add_main_op(self):
        if len(self._summary_ops) > 0:
            self.register_main_node(tf.summary.merge(self._summary_ops))

    def __create_writer(self, feeds):
        self.register_node('summary_writer',
                           tf.summary.FileWriter(
                               self.param('path', feeds),
                               get_default_session().graph))

    def __add_tensors_to_summary(self):
        for n, v in self._tensors.items():
            if v is None:
                continue
            op = None
            if v.stype == 'scalar':
                op = tf.summary.scalar(n, v.tensor)
            elif v.stype == 'image':
                op = tf.summary.image(n, v.tensor)
            if op is not None:
                self._summary_ops.append(op)


class SummaryWriterForNet(SummaryWriter):
    """
    A quick summary writer which automaticlly add loss, evaluate (if exists),
    input, inference of net.
    """

    def __init__(self,
                 name,
                 net,
                 tensors_to_summary=None,
                 inputs=None,
                 **config):
        automatic_summary_tensors = dict()
        automatic_keys = [NodeKeys.LOSS, NodeKeys.INFERENCE, NodeKeys.EVALUATE]
        for k in automatic_keys:
            if k in net.nodes:
                automatic_summary_tensors[k] = SummaryItem(net[k])
        if tensors_to_summary is None:
            tensors_to_summary = dict()
        super().__init__(name,
                         combine_dicts(tensors_to_summary,
                                       automatic_summary_tensors), inputs,
                         **config)


from dxpy.configs import configurable
from dxpy.learn.config import config


class SummaryWriterV2(Graph):
    # TODO: Target features:
    # 1. summary_switch, within summary content, summaries will be added, otherwise ignored.
    # 2. get_summary_writer(), global access to summary writer, convinient for adding summries
    # 3. graph_name_filter: only response to summaries from given graph paths
    @configurable(config, with_name=True)
    def __init__(self,
                 name='summary',
                 tensors_to_summary=None,
                 inputs=None,
                 *,
                 valid_paths=None,
                 additional_paths=None,
                 **kw):
        super().__init__(
            name,
            valid_paths=valid_paths,
            additional_paths=additional_paths,
            **kw)
