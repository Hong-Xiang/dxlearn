from dxl.learn.network.trainer import Trainer, RMSPropOptimizer
from .model import *
from .data import dataset_pytable, dataset_fast
import click
# from .data import create_fast_dataset

# from dxl.core.debug import profiled
from dxl.learn.core import Session
from dxl.learn.core.global_ctx import get_global_context

from dxl.learn.network.saver.saver import Saver
from dxl.learn.network.summary.summary import SummaryWriter


def cate_task_loss(infer, label):
    l = tf.losses.softmax_cross_entropy(label.data, infer.data)
    acc, acc_op = tf.metrics.accuracy(tf.argmax(infer.data, 1),
                                      tf.argmax(label.data, 1))
    return l, acc, acc_op


@click.command()
@click.option('--load', '-l', type=int)
@click.option('--path', '-p', type=click.types.Path(True, dir_okay=False))
@click.option('--steps', '-s', type=int, default=10000000)
@click.option('--nb-hits', '-n', type=int, default=2)
def train(path, load, steps, nb_hits):
    path_table = path
    save_path = './model'
    # padding_size = nb_hits
    padding_size = 5
    # d = create_dataset(dataset_pytable, path_db, padding_size, 128)
    # d = create_fast_dataset(path_db, 1024, True)
    # d_train = dataset_pytable(path_table, 128, True, nb_hits, True)
    # d_test = dataset_pytable(path_table, 128, True, nb_hits, False)
    batch_size = 2048
    is_with_coincidence = True
    d_train = dataset_fast(path_table, batch_size,
                           is_with_coincidence, nb_hits, True)
    d_test = dataset_fast(path_table, batch_size,
                          is_with_coincidence, nb_hits, False)
    # d = create_dataset(dataset_pytable, path_db, 128)
    get_global_context().make()
    model = IndexFirstHit(
        'model', max_nb_hits=padding_size, nb_units=[256] * 10)
    infer = model({'hits': d_train.hits})
    infer_test = model({'hits': d_test.hits})

    loss, train_acc, train_acc_op = cate_task_loss(
        infer, d_train.first_hit_index)
    _, test_acc, test_acc_op = cate_task_loss(
        infer_test, d_test.first_hit_index)

    t = Trainer('trainer', RMSPropOptimizer('opt', learning_rate=1e-4))
    t.make({'objective': loss})
    train_step = t.train_step
    saver = Saver('saver')
    # with profiled():
    with Session() as sess:
        sess.init()
        if load == 1:
            saver.restore(sess._raw_session, save_path)
        for i in range(steps):
            sess.run(train_step)
            if i % 100 == 0:
                loss_train, train_acc_v = sess.run([loss, train_acc_op])
                with get_global_context().test_phase():
                    test_acc_v = sess.run(test_acc_op)
                print("loss {}, train_acc {}, test_acc: {}".format(
                    loss_train, train_acc_v,  test_acc_v))
            if (i + 1) % 10000 == 0:
                saver.save(sess._raw_session, save_path)
