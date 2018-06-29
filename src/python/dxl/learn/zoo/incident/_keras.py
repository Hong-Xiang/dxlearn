from .data import HitsTable
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
import keras
from dxl.learn.function import OneHot
import numpy as np
import click


def load_pytable_dataset(path_h5):
    columns = HitsTable(path_h5)
    dtypes = {
        'hits': np.float32,
        'first_hit_index': np.int32,
        'padded_size': np.int32,
    }
    data = {k: np.array(columns.cache[k], dtype=dtypes[k])
            for k in columns.cache}
    columns.close()
    return data


import os


@click.command()
@click.option('--epochs', '-e', type=int, default=100)
@click.option('--load', '-l', type=int, default=0)
def train_keras(epochs, load):
    models = [Flatten(input_shape=(5, 4)), ]
    for i in range(5):
        models.append(Dense(100))
        models.append(Activation('relu'))
        models.append(Dropout(0.5))
    models.append(Dense(5))
    model = Sequential(models)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    path_h5 = os.environ['GHOME'] + \
        '/Workspace/IncidentEstimation/data/gamma.h5'
    data = load_pytable_dataset(path_h5)
    labels = keras.utils.to_categorical(data['first_hit_index'], num_classes=5)
    model_path = './model/kerasmodel-{}.h5'
    if load > 0:
        model.load_weights(model_path.format(load))
    for i in range(10):
        model.fit(data['hits'], labels, batch_size=128,
                  epochs=2, validation_split=0.2)
        model.save_weights(model_path.format(i))
