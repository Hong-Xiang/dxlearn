from dxl.learn.network.trainer import Trainer, RMSPropOptimizer
from .model import *
import click
from dxl.learn.tensor._global_step import GlobalStep
from dxl.learn.core import Session
from dxl.learn.core.global_ctx import get_global_context

from dxl.learn.network.saver.saver import Saver


from dxl.function import fmap, to
from dxl.data.function import x


from dxl.learn.zoo.incident.main import construct_network, make_summary, get_saver

from dxl.learn.zoo.incident.main import one_hot_predict, same_crystal_accuracy

from typing import NamedTuple, List

from dxl.learn.train import TrainSpec
import json


@click.command()
@click.option('--config-file', '-c', type=click.types.Path(True, dir_okay=False))
def train(config_file):
    with open(config_file) as fin:
        config = json.load(fin)
    spec = TrainSpec.from_dict(config)
    path_table = config['path_table']
    nb_hits = config['nb_hits']
    result_train, result_test = construct_network(path_table, None, nb_hits, spec)
    loss_train = result_train['loss']
    t = Trainer('trainer', RMSPropOptimizer('opt', learning_rate=spec.learning_rate))

    t.make({'objective': loss_train})
    train_step = t.train_step
    saver = get_saver(spec)
    sw_train, sw_test = make_summary(result_train, 'train', spec), make_summary(result_test, 'test', spec)

    fetches = {
        'loss_train': result_train['loss'],
        'acc_train': result_train['accuracy'],
        'loss_test': result_test['loss'],
        'acc_test': result_test['accuracy']
    }
    with Session() as sess:
        sess.init()
        if spec.load_step == 1:
            saver.restore(sess._raw_session, save_path)
        for i in range(spec.nb_steps):
            sess.run(train_step)
            # if i  % 1000 == 0:
                # print("step {}".format(i))
            if i % max(spec.nb_steps // 10000, 1) == 0:
                with get_global_context().test_phase():
                    fetched = sess.run(fetches)
                    print(fetched)
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