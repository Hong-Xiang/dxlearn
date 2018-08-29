import tensorflow as tf
from fs.osfs import OSFS


def dummy_image(shape=None, batch_size=32, nb_channel=1, name='dummy_image'):
    if shape is None:
        shape = [28, 28]
    return tf.placeholder(
        tf.float32, [batch_size] + shape + [nb_channel], name=name)


# def write_graph(path=None):
#     from ..train.summary import SummaryWriter
#     from ..depsession import Session
#     if path is None:
#         path = '/tmp/tf_tests'
#     with OSFS(path) as fs:
#         dir_name = path
#         if fs.exists(dir_name):
#             d = fs.opendir(dir_name)
#         else:
#             d = fs.makedir(dir_name)
#         if d.exists('summary/train'):
#             d.removetree('summary/train')
#         summary = SummaryWriter(
#             name='train', path=d.getsyspath('summary/train'))
#         depsession = Session()
#         with depsession.as_default():
#             summary.post_session_created()


def write_graph(path=None):
    if path is None:
        path = '/tmp/tf_tests'
    with OSFS(path) as fs:
        dir_name = 'summary'
        if fs.exists(dir_name):
            d = fs.opendir(dir_name)
        else:
            d = fs.makedir(dir_name)
        if d.exists('summary/train'):
            d.removetree('summary/train')
        with tf.Session() as sess:
            summary = tf.summary.FileWriter(
                d.getsyspath('summary/train'), sess.graph)
