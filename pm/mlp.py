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
feature_length = read.window_length * read.frames_per_second * read.frame_size
k = 3


def mlp():
    _input = Input(shape=(feature_length,))
    x = Dense(1200, activation='relu')(_input)
    x = BatchNormalization()(x)
    return Model(inputs=_input, outputs=x, name='embedding')


all_features = read.read()
test_ids = list(all_features.keys())

for test_id in test_ids:
    _train_features, _test_features = read.split(all_features, test_id)

    _train_features, _train_labels = read.flatten(_train_features)
    _test_features, _test_labels = read.flatten(_test_features)

    _train_labels_ = np_utils.to_categorical(_train_labels, len(read.activity_list))
    _test_labels_ = np_utils.to_categorical(_test_labels, len(read.activity_list))

    _train_features = np.array(_train_features)
    _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1] * _train_features.shape[2]))
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.reshape(_test_features, (_test_features.shape[0], _test_features.shape[1] * _test_features.shape[2]))
    print(_test_features.shape)

    _input_ = Input(shape=(feature_length,))
    base_network = mlp()
    base = base_network(_input_)
    classifier = Dense(len(read.activity_list), activation='softmax')(base)
    _model = Model(inputs=_input_, outputs=classifier, name='classifier')

    _model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    _model.fit(_train_features, _train_labels_, verbose=1, batch_size=batch_size, epochs=epochs, shuffle=True)

    _train_preds = base_network.predict(_train_features)
    _test_preds = base_network.predict(_test_features)

    # classifier evaluation
    results = _model.evaluate(_test_features, _test_labels_, batch_size=batch_size, verbose=0)
    print(results)

    # knn evaluation
    acc = read.cos_knn(k, _test_preds, _test_labels, _train_preds, _train_labels)
    read.write_data('mlp.csv', 'score:'+','.join([str(f) for f in results])+',knn_acc,'+str(acc))
