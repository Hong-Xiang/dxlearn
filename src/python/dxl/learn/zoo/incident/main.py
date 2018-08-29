from dxl.function import fmap
from dxl.data.function import x
from dxl.learn.function import OneHot, shape_list
from .data import (load_table, SplitByPeriod, binary_crystal_index,
                   parse_hits_features, parse_crystal_index)

import tensorflow as tf
import numpy as np
from .model import FirstHit
from dxl.learn.core.global_ctx import get_global_context
from dxl.learn.core import Tensor
from dxl.learn.network.summary.summary import SummaryWriter, ScalarSummary

from dxl.learn.function import Sum, ArgMax, OneHot

from dxl.learn.network.saver.saver import Saver


def dataset_photon_classification(path_h5, limit, nb_hits):
    photons = load_table(path_h5, limit)
    photons = [p for p in photons if p.nb_true_hits == nb_hits]
    photons = [p.update(hits=p.hits[:nb_hits]) for p in photons]
    photons = fmap(binary_crystal_index, photons)
    train, test = SplitByPeriod(10, list(range(8)))(photons)
    return train, test


def fetch_features(samples):
    return fmap(parse_hits_features, samples), fmap(parse_crystal_index, samples)


def make_dataset(samples):
    return tf.data.Dataset.from_tensor_slices(np.array(samples, dtype=np.float32))


def combine_feature_datasets(hits, crystal_index, spec):
    return (tf.data.Dataset.zip({'hits': hits, 'crystal_index': crystal_index})
            .repeat()
            .shuffle(spec.batch_size*8)
            .batch(spec.batch_size)
            .make_one_shot_iterator()
            .get_next())


def make_one_dataset(samples, spec):
    hits, crystal_index = fetch_features(samples)
    hits, crystal_index = make_dataset(hits), make_dataset(crystal_index)
    return combine_feature_datasets(hits, crystal_index, spec)


def prepare_all_datasets(path_h5, limit, nb_hits, spec):
    train, test = dataset_photon_classification(path_h5, limit, nb_hits)
    train = make_one_dataset(train, spec)
    test = make_one_dataset(test, spec)
    return train, test


def apply_model(model, dataset, spec):
    infer = model({'hits': Tensor(dataset['hits'])})
    label = dataset['crystal_index']
    loss = tf.losses.sigmoid_cross_entropy(label, infer.data)
    acc = same_crystal_accuracy(label, infer.data, spec)
    return {"infer": infer, "loss": loss, "accuracy": acc, 'label': label}


def construct_network(path_h5, limit, nb_hits, spec):
    get_global_context().make()
    dataset_train, dataset_test = prepare_all_datasets(
        path_h5, limit, nb_hits, spec)
    model = FirstHit('model', max_nb_hits=nb_hits,
                     nb_units=spec.nb_units)
    results_train = apply_model(model, dataset_train, spec)
    results_test = apply_model(model, dataset_test, spec)
    return results_train, results_test


def make_summary(result, name, spec):
    sw = SummaryWriter('sw_{}'.format(name),
                       './summary/{}'.format(name))
    sw.add_graph()
    sw.add_item(ScalarSummary('loss', result['loss']))
    sw.add_item(ScalarSummary('accuracy', result['accuracy']))
    sw.make()
    return sw


def one_hot_predict(predict):
    return OneHot(shape_list(predict)[1])(ArgMax(1)(predict))


def same_crystal_accuracy(crystal_indices, predict, spec):
    mask = one_hot_predict(predict) * crystal_indices
    accuracy = Sum()(mask) / spec.batch_size
    return accuracy


def get_saver(spec):
    saver = Saver('saver', save_interval=3000,
                  model_dir='./model')
    saver.make()
    return saver
