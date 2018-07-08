from dxl.learn.network.trainer import Trainer, RMSPropOptimizer
from .model import *
import click
from dxl.learn.tensor.global_step import GlobalStep
from dxl.learn.core import Session
from dxl.learn.core.global_ctx import get_global_context

from dxl.learn.network.saver.saver import Saver


from dxl.function import fmap, to
from dxl.data.function import x


from dxl.learn.zoo.incident.main import construct_network, make_summary

from dxl.learn.zoo.incident.main import one_hot_predict, same_crystal_accuracy


@click.command()
@click.option('--load', '-l', type=int)
@click.option('--path', '-p', type=click.types.Path(True, dir_okay=False))
@click.option('--steps', '-s', type=int, default=10000000)
@click.option('--nb-hits', '-n', type=int, default=2)
def train(path, load, steps, nb_hits):
    path_table = path
    save_path = './model'
    # padding_size = nb_hits
    result_train, result_test = construct_network(path, None, nb_hits)
    loss_train = result_train['loss']
    t = Trainer('trainer', RMSPropOptimizer('opt', learning_rate=1e-3))

    t.make({'objective': loss_train})
    train_step = t.train_step
    saver = Saver('saver', save_interval=30)
    saver.make()
    sw_train, sw_test = make_summary(
        result_train, 'train'), make_summary(result_test, 'test')

    fetches = {
        'loss_train': result_train['loss'],
        'acc_train': result_train['accuracy'],
        'loss_test': result_test['loss'],
        'acc_test': result_test['accuracy']
    }
    # with profiled():
    with Session() as sess:
        sess.init()
        # debug(sess, result_train) 
        # return
        if load == 1:
            saver.restore(sess._raw_session, save_path)
        for i in range(steps):
            sess.run(train_step)
            if i % 100 == 0:
                # loss_train, train_acc_v = sess.run([loss, train_acc_op])
                fetched = sess.run(fetches)
                print(fetched)
                # with get_global_context().test_phase():
                #     test_acc_v = sess.run(test_acc_op)
                # print("loss {}, train_acc {}, test_acc: {}".format(
                #     loss_train, train_acc_v,  test_acc_v))
                sw_train.run()
                sw_test.run()
            saver.auto_save()
    sw.close()

def debug(sess, result_train):
    print('infer')
    print(sess.run(result_train['infer']))
    print('label')
    print(sess.run(result_train['label']))
    print('one_hot_pred')
    print(sess.run(one_hot_predict(result_train['infer'].data)))
    print(sess.run(same_crystal_accuracy(result_train['label'],
                                            result_train['infer'].data)))
    return