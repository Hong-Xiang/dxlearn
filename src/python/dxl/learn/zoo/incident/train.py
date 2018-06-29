from dxl.learn.network.trainer import Trainer, RMSPropOptimizer
from .model import *
from .data import create_fast_dataset

from dxl.core.debug import profiled


@click.command()
@click.option('--load', '-l', type=int)
@click.option('--steps', '-s', type=int, default=10000000)
def train(load, steps):
    path_db = os.environ['GHOME'] + \
        '/Workspace/IncidentEstimation/data/gamma.h5'
    save_path = './model'
    padding_size = 5
    # d = create_dataset(dataset_pytable, path_db, padding_size, 128)
    d = create_fast_dataset(path_db, 1024, True)
    # d = create_dataset(dataset_pytable, path_db, 128)
    model = IndexFirstHit('model', d.hits, padding_size, [100] * 5)
    infer = model()

    l = tf.losses.softmax_cross_entropy(d.first_hit_index.data, infer.data)
    acc, acc_op = tf.metrics.accuracy(tf.argmax(d.first_hit_index.data, 1),
                                      tf.argmax(infer.data, 1))

    t = Trainer('trainer', RMSPropOptimizer('opt', learning_rate=1e-3))
    t.make({'objective': l})
    train_step = t.train_step
    saver = tf.train.Saver()
    # with profiled():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if load == 1:
            saver.restore(sess, save_path)
        for i in range(steps):
            sess.run(train_step)
            if i % 100 == 0:
                print(sess.run([acc, acc_op]))
                print(sess.run([l, acc]))
            if (i+1) % 10000 == 0:
                saver.save(sess, save_path)

