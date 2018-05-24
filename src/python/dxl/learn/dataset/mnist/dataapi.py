import tensorflow as tf
from ..base import DatasetTFRecords, NodeKeys


class MNISTTFRecords(DatasetTFRecords):
    def __init__(self, name='dataset', *, files=None, batch_size=None, one_hot=None, **config):
        super(__class__, self).__init__(name, files=files,
                                        batch_size=batch_size,
                                        one_hot=one_hot,
                                        **config)

    @classmethod
    def _default_config(cls):
        return {
            'files': ['/home/hongxwing/Datas/mnist/mnist.train.tfrecord'],
            'batch_size': 32,
            'one_hot': True,
        }

    @classmethod
    def _parse_example(cls, example):
        return tf.parse_single_example(example, features={
            'image': tf.FixedLenFeature([], tf.string),
            'shape': tf.FixedLenFeature([2], tf.int64),
            'label': tf.FixedLenFeature([1], tf.int64)
        })

    @classmethod
    def _decode_image(cls, data):
        return {'image': tf.decode_raw(data['image'], tf.uint8),
                'shape': data['shape'],
                'label': data['label']}

    @classmethod
    def _change_dtype(cls, data):
        return {'image': tf.to_float(data['image']),
                'shape': data['shape'],
                'label': data['label']}

    @classmethod
    def _reshape_tensors(cls, data):
        return {'image': tf.reshape(data['image'], [28, 28, 1]),
                'shape': tf.concat([data['shape'], [1]], axis=0),
                'label': tf.reshape(data['label'], [])}

    def _normalize(self, data):
        if self._normalizer is None:
            image = data['image']
        else:
            image = self._normalizer(data['image'])[NodeKeys.MAIN]
        return {'image': image,
                'shape': data['shape'],
                'label': data['label']}
        return data

    def _maybe_onehot(self, data):
        result = {
            'image': data['image'],
            'shape': data['shape']
        }
        if self.c['one_hot']:
            result.update({'label': tf.one_hot(data['label'], depth=10)})
        else:
            result.update({'label': data['label']})
        return result

    def _processing(self):
        dataset = super()._processing()
        return (dataset
                .map(self._parse_example)
                .map(self._decode_image)
                .map(self._change_dtype)
                .map(self._reshape_tensors)
                # .map(self._normalize)
                .map(self._maybe_onehot)
                .batch(self.c['batch_size'])
                .cache()
                .repeat())


# from ..base import Graph


# class MNISTLoadAll(Graph):
#     def __init__(self, name='/dataset', **config):
#         # from ..preprocessing.normalizer import SelfMinMax

#         super(__class__, self).__init__(name, **config)
#         self._normalizer = None
#         method = self.c['normalization']['method'].lower()
#         if method != 'pass':
#             self._normalizer = get_normalizer(
#                 method)(self.name / 'normalization', )
#         self.register_main_node(self.__load_data())
#         self.register_node('image', self.as_tensor()['image'])
#         self.register_node('label', self.as_tensor()['label'])

#     def get_feed_dict(self):
#         return tf.get_default_session().run(
#             {'image': self.tensor('image'), 'label': self.tensor('label')})

#     @classmethod
#     def _default_config(cls):
#         return {
#             'batch_size': 32,
#         }

#     def post_session_created(self):
#         tf.get_default_session().run(self._iter_init,
#                                      feed_dict={
#                                          self.tensor('images'): self.images,
#                                          self.tensor('labels'): self.labels
#                                      })

#     def _normalize(self, image):
#         if self._normalizer is None:
#             return image
#         else:
#             return self._normalizer(image)['data']

#     def __load_data(self):
#         from tensorflow.examples.tutorials.mnist import input_data
#         from tensorflow.contrib.data import Dataset
#         mnist = input_data.read_data_sets(
#             '/home/hongxwing/Datas/mnist/tfdefault', one_hot=True)
#         self.images = mnist.train.images
#         self.labels = mnist.train.labels

#         # def gen_image():
#         #     for i in range(images.shape[0]):
#         #         yield {'image': images[i, ...], 'label': label[i, ...]}
#         with tf.name_scope(self.basename):
#             images_tensor = tf.placeholder(tf.float32, self.images.shape,
#                                            'mnist_images')
#             self.register_node('images', images_tensor)
#             labels_tensor = tf.placeholder(tf.int64, self.labels.shape,
#                                            'mnist_label')
#             self.register_node('labels', labels_tensor)
#             dataset_image = (Dataset.from_tensor_slices(images_tensor)
#                              .map(lambda img: tf.reshape(img, [28, 28, 1]))
#                              .map(self._normalize))
#             dataset_label = Dataset.from_tensor_slices(labels_tensor)
#             dataset = Dataset.zip(
#                 {'image': dataset_image, 'label': dataset_label})
#             dataset = dataset.batch(self.param('batch_size')).repeat()
#             iterator = dataset.make_initializable_iterator()
#             self._iter_init = iterator.initializer
#             next_element = iterator.get_next()
#         return next_element
