import tensorflow as tf
from ..model import Model
from ..net import Net
from ..graph import Graph, NodeKeys


class MNISTSimpleConvModel(Model):
    def __init__(self, name='/model', image=None, label=None, **config):
        inputs = {'image': image, 'label': label}
        super(__class__, self).__init__(name, inputs, **config)

    @classmethod
    def _default_inputs(self):
        return {
            'image': {'shape': [None, 28, 28, 1]},
            'label': {'shape': [None, 10]}}

    def _kernel(self, feeds):
        x = feeds['image']
        result = dict()
        with tf.variable_scope('kernel'):
            x = tf.layers.conv2d(x, 32, 5, padding='same',
                                 name='conv1', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 32, 1, padding='same',
                                 name='pool1', strides=(2, 2), activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 3, padding='same',
                                 name='conv2', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 1, padding='same',
                                 name='pool2', strides=(2, 2), activation=tf.nn.relu)
            with tf.name_scope('flatten'):
                x = tf.reshape(x, [-1, 7 * 7 * 64])
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='fc1')
            result['logits'] = tf.layers.dense(x, 10, name='fc2')
        with tf.name_scope('prediction'):
            result['prediction'] = tf.argmax(result['logits'], 1)
        if 'label' in feeds:
            with tf.name_scope('loss'):
                result['cross_entropy'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=feeds['label'], logits=result['logits']))
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(
                    result['prediction'], tf.argmax(feeds['label'], 1))
                result['accuracy'] = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))
        return result


class MNISTSimpleNet(Net):
    def __init__(self, name='/net', image=None, label=None, **config):
        inputs = {'image': image, 'label': label}
        super(__class__, self).__init__(name, inputs, **config)

    @classmethod
    def _default_inputs(self):
        return {
            'image': {'shape': [None, 28, 28, 1]},
            'label': {'shape': [None, 10]}}

    def _tensors_need_summary(self):
        return {
            'loss': {
                'type': 'scalar',
                'tensor': self.tensor(NodeKeys.LOSS)
            },
            'accuracy': {
                'type': 'scalar',
                'tensor': self.tensor(NodeKeys.EVALUATE)
            },
            'image': {
                'type': 'image',
                'tensor': self.tensor('image')
            }
        }

    def _pre_kernel_post_inputs(self):

        self._main_model = MNISTSimpleConvModel(self.name / 'model',
                                                self.nodes['image'],
                                                self.nodes['label'], lazy_create=True)

    def _post_kernel_post_outputs(self):
        self.register_node(
            NodeKeys.LOSS, self._main_model['cross_entropy'])
        self.register_node(NodeKeys.INFERENCE,
                           self._main_model['prediction'])
        self.register_node(NodeKeys.EVALUATE,
                           self._main_model['accuracy'])
        super()._post_kernel_post_outputs()

    def _kernel(self, feeds=None):
        return self._main_model({'image': self.tensor('image'), 'label': self.tensor('label')})


def main():
    from dxpy.learn.dataset.mnist import MNISTLoadAll
    from dxpy.learn.train.summary_writer import SummaryWriter
    from dxpy.learn.scalar import create_global_scalars
    create_global_scalars()
    create_global_scalars()
    dataset = MNISTLoadAll()
    net = MNISTSimpleNet(
        '/net', image=dataset['image'], label=dataset['label'])
    summary = SummaryWriter(
        'train', net._tensors_need_summary(), path='./summary/train/')
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        dataset.post_session_created()
        net.post_session_created()
        summary.post_session_created()
        feeds = sess.run(
            {'image': dataset['image'], 'label': dataset['label']})
        print('initial accuracy', net.evaluate())
        for i in tqdm_notebook(range(10)):
            for _ in tqdm_notebook(range(100)):
                net.train()
            feeds = dataset.get_feed_dict()
            print('STEP={}'.format((i + 1) * 100), net.evaluate())
            summary.summary()
            net.save()
        summary.flush()
