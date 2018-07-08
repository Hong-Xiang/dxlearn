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

BATCH_SIZE = 128

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


def combine_feature_datasets(hits, crystal_index):
    return (tf.data.Dataset.zip({'hits': hits, 'crystal_index': crystal_index})
            .repeat()
            .shuffle(1024)
            .batch(BATCH_SIZE)
            .make_one_shot_iterator()
            .get_next())


def make_one_dataset(samples):
    hits, crystal_index = fetch_features(samples)
    hits, crystal_index = make_dataset(hits), make_dataset(crystal_index)
    return combine_feature_datasets(hits, crystal_index)


def prepare_all_datasets(path_h5, limit, nb_hits):
    train, test = dataset_photon_classification(path_h5, limit, nb_hits)
    train, test = make_one_dataset(train), make_one_dataset(test)
    return train, test


def apply_model(model, dataset):
    infer = model({'hits': Tensor(dataset['hits'])})
    label = dataset['crystal_index']
    loss = tf.losses.sigmoid_cross_entropy(label, infer.data)
    acc = same_crystal_accuracy(label, infer.data)
    return {"infer": infer, "loss": loss, "accuracy": acc, 'label': label}


def construct_network(path_h5, limit, nb_hits):
    get_global_context().make()
    dataset_train, dataset_test = prepare_all_datasets(path_h5, limit, nb_hits)
    model = FirstHit('model', max_nb_hits=2, nb_units=[128]*5)
    results_train = apply_model(model, dataset_train)
    results_test = apply_model(model, dataset_test)
    return results_train, results_test

def make_summary(result, name):
    sw = SummaryWriter('sw_{}'.format(name), './summary/{}'.format(name))
    sw.add_graph()
    sw.add_item(ScalarSummary('loss', result['loss']))
    sw.add_item(ScalarSummary('accuracy', result['accuracy']))
    sw.make()
    return sw

def one_hot_predict(predict):
    return OneHot(shape_list(predict)[1])(ArgMax(1)(predict))
    
def same_crystal_accuracy(crystal_indices, predict):
    mask = one_hot_predict(predict) * crystal_indices
    accuracy = Sum()(mask) / BATCH_SIZE
    return accuracy