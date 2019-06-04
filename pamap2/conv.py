import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import read
import sys
from keras.utils import np_utils

np.random.seed(1)
tf.set_random_seed(2)

feature_length = read.dct_length * 3 * 3
batch_size = 60
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
    _input = Input(shape=(feature_length, 1))
    x = Conv1D(12, kernel_size=3, activation='relu')(_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(len(read.classes), activation='softmax')(x)
    return Model(inputs=_input, outputs=x)


feature_data = read.read()

test_ids = list(feature_data.keys())
test_id = [test_ids[sys.argv[1]]]

_train_data, _test_data = split(feature_data, test_id)
_train_data, _train_labels = flatten(_train_data)
_test_data, _test_labels = flatten(_test_data)

_train_data = np.array(_train_data)
_train_data = np.expand_dims(_train_data, 3)

_test_data = np.array(_test_data)
_test_data = np.expand_dims(_test_data, 3)

_test_labels = np_utils.to_categorical(_test_labels, len(read.classes))
_train_labels = np_utils.to_categorical(_train_labels, len(read.classes))

model = conv()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(_train_data, _train_labels, epochs=epochs, batch_size=batch_size, verbose=1)
score = model.evaluate(_test_data, _test_labels, batch_size=batch_size, verbose=1)

print(score)
read.write_data('conv architecture' + ',' + 'window_length:' + str(read.window_length) + ',' + 'dct_length:' + str(
    read.dct_length) + ',' + 'increment_ratio:' + str(read.increment_ratio) + ',' + 'batch_size:' + str(batch_size)
                + ',' + 'epochs:' + str(epochs) + ',' + 'test_id:' + str(
    test_id[0]) + ',' + 'score:' + ','.join([str(f) for f in score]))
