import numpy as np
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.models import Model
from tensorflow import set_random_seed
import os

import random
import sys
import read

random.seed(0)
np.random.seed(1)
set_random_seed(2)

batch_size = 64
epochs = 10


def split(_data, _test_ids):
    train_data_ = {key: value for key, value in _data.items() if key not in _test_ids}
    test_data_ = {key: value for key, value in _data.items() if key in _test_ids}
    return train_data_, test_data_


def flatten(_data):
    flatten_data = []
    flatten_labels = []

    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_data.extend(activity_data)
            flatten_labels.extend([activity for i in range(len(activity_data))])
    return flatten_data, flatten_labels


def conv():
    _input = Input(shape=(read.dct_length*3*len(read.sensors),1))
    x = Conv1D(32, kernel_size=5, activation='relu')(_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(len(read.activity_list), activation='softmax')(x)

    model = Model(inputs=_input, outputs=x)
    model.summary()
    return model


all_features = read.read()
test_ids = list(all_features.keys())
test_id = [test_ids[0]]#[test_ids[sys.argv[1]]]

_train_features, _test_features = split(all_features, test_id)

_train_features, _train_labels = flatten(_train_features)
_test_features, _test_labels = flatten(_test_features)

_train_labels = np_utils.to_categorical(_train_labels, len(read.activity_list))
_test_labels = np_utils.to_categorical(_test_labels, len(read.activity_list))

_train_features = np.array(_train_features)
_train_features = np.expand_dims(_train_features, 3)
print(_train_features.shape)

_test_features = np.array(_test_features)
_test_features = np.expand_dims(_test_features, 3)
print(_test_features.shape)

_model = conv()
_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
_model.fit(_train_features, _train_labels, verbose=1, batch_size=batch_size, epochs=epochs, shuffle=True)
results = _model.evaluate(_test_features, _test_labels, batch_size=batch_size, verbose=0)
print(results)
read.write_data('conv architecture' + ',' + 'window_length:' + str(read.window_length) + ',' + 'dct_length:' + str(
    read.dct_length) + ',' + 'increment_ratio:' + str(read.increment_ratio) + ',' + 'batch_size:' + str(
    batch_size) + ',' + 'epochs:' + str(epochs) + ',' + 'test_id:' + str(test_id[0]) + ',' + 'score:' + ','.join(
    [str(f) for f in results]))
