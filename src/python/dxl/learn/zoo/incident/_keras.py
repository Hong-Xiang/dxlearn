from .data import ShuffledHitsTable
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
import keras
from dxl.learn.function import OneHot
import numpy as np
import click
from dxl.data.zoo.incident_position_estimation.function import filter_by_nb_hits


def load_pytable_dataset(path_h5):
    columns = ShuffledHitsTable(path_h5)
    columns = filter_by_nb_hits(columns, 2)
    dtypes = {
        'hits': np.float32,
        'first_hit_index': np.int32,
        'padded_size': np.int32,
    }
    data = {}
    for k in dtypes:
        data[k] = np.array([getattr(d, k)
                            for d in columns.data], dtype=dtypes[k])
    # columns.close()
    return data


import os


@click.command()
@click.option('--epochs', '-e', type=int, default=100)
@click.option('--load', '-l', type=int, default=0)
@click.option('--path-data', '-p', type=click.types.Path(exists=True, dir_okay=False))
def train_keras(epochs, load, path_data):
    data = load_pytable_dataset(path_data)
    padding_size = data['hits'].shape[1]
    nb_features = data['hits'].shape[2]
    print('Padding size: {}, nb features: {}.'.format(padding_size, nb_features))

    models = [Flatten(input_shape=(padding_size, nb_features))]
    for i in range(5):
        models.append(Dense(256))
        models.append(Activation('relu'))
        models.append(Dropout(0.5))
    models.append(Dense(padding_size))
    model = Sequential(models)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    labels = keras.utils.to_categorical(data['first_hit_index'], num_classes=5)
    model_path = './model/kerasmodel-{}.h5'
    if load > 0:
        model.load_weights(model_path.format(load))
    for i in range(10):
        model.fit(data['hits'], labels, batch_size=1024,
                  epochs=epochs // 10, validation_split=0.01)
        model.save_weights(model_path.format(i))
