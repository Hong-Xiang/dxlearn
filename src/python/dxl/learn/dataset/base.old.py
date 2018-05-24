import numpy as np
import tensorflow as tf
from ..graph import Graph, NodeKeys

from dxpy.configs import configurable
from ..config import get_config, config


class DatasetClassic(Graph):
    """Base class of database
    Database add some special tasks based on graph:
        1. single
        2. batch
    These methods returns a dict of Tensors:


    Parameters:
        batch_size: int, batch size of samples
        fields:
            key, shape
    """

    def __init__(self, name, **config):
        super(__class__, self).__init__(name, **config)
        with tf.name_scope(self.basename):
            for f in self.param('fields'):
                field_config = self.param('fields')[f]
                self.create_placeholder_node(
                    field_config['dtype'], self.batched_shape(f), f)
        self.register_task('single', self.single)
        self.register_task('batch', self.batch)

    def _load_sample(self, feeds=None):
        raise NotImplementedError

    def _load_dummpy(self, feeds=None):
        raise NotImplementedError

    def get_feed_dict(self, task=None):
        result = dict()
        batched_data = self.batch()
        for f in self.param('fields'):
            result[self.nodes[f]] = batched_data[f]
        return result

    def single(self, feeds=None):
        try:
            return self._load_sample(feeds)
        except StopIteration:
            return self._load_dummpy(feeds)

    def batch(self, feeds=None):
        result = dict()
        for f in self.param('fields', feeds):
            result[f] = np.zeros(self.batched_shape(f))
        for i in range(self.param('batch_size', feeds)):
            next_sample = self.single(feeds)
            for k in next_sample:
                result[k][i, ...] = next_sample[k]
        return result

    def batched_shape(self, field_name):
        return list([self.param('batch_size')] +
                    list(self.param('fields')[field_name]['shape']))


class DatasetBase(Graph):
    def __init__(self, name, **config):
        super().__init__(name, **config)
        self._pre_processing()
        with tf.name_scope(self.basename):
            self.dataset = self._processing()
            self.__register_dataset()
            self._post_processing()

    def _pre_processing(self):
        pass

    def _post_processing(self):
        # for k in self.as_tensor():
        pass

    def post_session_created(self):
        pass

    def _processing(self):
        raise NotImplementedError

    def __register_dataset(self):
        iterator = self.dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        self.register_main_node(next_element)
        for k in next_element:
            self.register_node(k, next_element[k])


class DatasetTFRecords(DatasetBase):
    def __init__(self, name, **config):
        super().__init__(name, **config)

    def _load_tfrecord_files(self):
        from dxpy.batch import FilesFilter
        from fs.osfs import OSFS
        return tf.data.TFRecordDataset(self.param('files'))

    def _processing(self):
        return self._load_tfrecord_files()


class DatasetBasev2(Graph):
    @configurable(get_config(), with_name=True)
    def __init__(self, name, batch_size=8, **config):
        super().__init__(name, batch_size=batch_size, **config)
        self._pre_processing()
        with tf.name_scope(self.basename):
            self.dataset = self._processing()
            self.__register_dataset()
            self._post_processing()

    def _pre_processing(self):
        pass

    def _post_processing(self):
        from dxpy.learn.utils.tensor import ensure_shape
        for k in self.as_tensor():
            self.nodes[k] = ensure_shape(
                self.nodes[k], batch_size=self.param('batch_size'))

    def post_session_created(self):
        pass

    def _processing(self):
        raise NotImplementedError

    def __register_dataset(self):
        iterator = self.dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        self.register_main_node(next_element)
        for k in next_element:
            self.register_node(k, next_element[k])


class DatasetFromTFDataset(DatasetBasev2):
    @configurable(get_config(), with_name=True)
    def __init__(self, name, tf_dataset, **config):
        self.tf_dataset = tf_dataset
        super().__init__(name, **config)

    def _processing(self):
        return self.tf_dataset

class DatasetFromGenerator(DatasetBasev2):
    @configurable(config, with_name=True)
    def __init__(self, name, generator, output_types, output_shapes, **config):
        self.generator = generator
        super().__init__(name, output_types=output_types, output_shapes=output_shapes, **config)

    def _processing(self):
        return tf.data.Dataset.from_generator(self.generator, self.param('output_types'), self.param('output_shapes'))

