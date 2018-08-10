import tensorflow as tf
from ..net import Net
from ..graph import NodeKeys


class SRStackCNN(Net):
    def __init__(self, name, low_image, high_image, **config):
        super().__init__(name,
                         inputs={NodeKeys.INPUT: low_image,
                                 NodeKeys.LABEL: high_image}, **config)

    @classmethod
    def _default_config(cls):
        from dxpy.collections.dicts import combine_dicts
        cfg = {
            'input_shape': [14, 14, 1],
            'label_shape': [28, 28, 1]
        }
        return combine_dicts(cfg, super()._default_config())

    @classmethod
    def _default_inputs(cls):
        return {
            NodeKeys.INPUT: {'shape': cls._default_config()['input_shape']},
            NodeKeys.LABEL: {'shape': cls._default_config()['label_shape']}
        }

    def _pre_kernel_post_inputs(self):
        from ..model.cnn.blocks import StackedConv2D
        model = StackedConv2D('stackedcnns', self.tensor(
            NodeKeys.INPUT), 10, lazy_create=True)
        self.register_child_model('sr_kernel', model)

    def _post_kernel_post_outputs(self):
        super()._post_kernel_post_outputs()
        self.register_node(NodeKeys.EVALUATE, self.tensor(NodeKeys.LOSS))

    def _kernel(self, feeds):
        from ..model.image import resize
        x = feeds[NodeKeys.INPUT]
        u = resize(x, (2, 2))
        h = self.child_model('sr_kernel')(u)
        with tf.name_scope('inference'):
            y = tf.layers.conv2d(h, 1, 3, padding='same')
        result = {NodeKeys.INFERENCE: y}
        if NodeKeys.LABEL in feeds:
            with tf.name_scope('loss'):
                l = tf.losses.mean_squared_error(feeds[NodeKeys.LABEL], y)
        result[NodeKeys.LOSS] = l
        return result





def main():
    import numpy as np
    import tensorflow as tf
    from tqdm import tqdm
    from dxpy.learn.config import config
    from dxpy.learn.scalar import create_global_scalars
    from dxpy.learn.graph import NodeKeys
    from dxpy.learn.train.summary_writer import SummaryWriter, SummaryItem
    from dxpy.learn.session import Session
    from dxpy.learn.model.tensor import PlaceHolder
    from dxpy.learn.dataset.petsino import PhantomSinograms
    from dxpy.learn.dataset.mnist import MNISTTFRecords
    from dxpy.learn.zoo.super_resolution import SRStackCNN
    from dxpy.learn.dataset.super_resolution import SuperResolutionDataset
    from dxpy.numpy_extension.visual.image import grid_view

    files = [
        '/home/hongxwing/Datas/phantom/phantom.{}.tfrecord'.format(i) for i in range(10)]
    config_sinogram = {
        'batch_size': 8,
        'files': files,
    }
    config_mnist = {
        'batch_size': 32,
        'normalization': {
            'method': 'selfminmax'
        }
    }
    config['dataset'] = {
        'sinograms': config_sinogram,
        'mnist': config_mnist
    }
    config_stacked_cnns = {
        'nb_layers': 10,
    }
    config['network'] = {
        'activation': 'relu',
        'stackedcnns': config_stacked_cnns,
    }
    create_global_scalars()

    dataset = SuperResolutionDataset('dataset', lambda: MNISTTFRecords(
        'dataset/mnist'), input_key='image', nb_down_sample=1)
    network = SRStackCNN('network', dataset['image2x'], dataset['image1x'])
    summary_loss = SummaryItem(network[NodeKeys.LOSS], 'scalar')
    summary = SummaryWriter(
        name='summary/train', tensors_to_summary={'loss': summary_loss}, path='./summary/train/')
    session = Session()
    with session.as_tensor().as_default():
        session.post_session_created()
        network.post_session_created()
        summary.post_session_created()
    with session.as_default():
        for i in range(2000):
            network.train()
            if i % 100 == 0:
                summary.summary()
        network.nodes[NodeKeys.TRAINER].decay_learning_rate(0.1)
        for i in range(2000):
            network.train()
            if i % 100 == 0:
                summary.summary()

    with session.as_default():
        imginf = network.inference(feeds={NodeKeys.INPUT: img2x})
