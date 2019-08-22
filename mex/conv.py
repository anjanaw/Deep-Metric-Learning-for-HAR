import numpy as np
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.models import Model
from tensorflow import set_random_seed
import os
import heapq
from sklearn.metrics.pairwise import cosine_similarity
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


def cos_knn(k, test_data, test_labels, stored_data, stored_target):
    cosim = cosine_similarity(test_data, stored_data)

    top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]
    top = [[stored_target[j] for j in i[:k]] for i in top]

    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)

    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == pred[j]:
            correct += 1
    return correct / float(len(test_labels))


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
    _input = Input(shape=(read.dct_length*3*len(read.sensors), 1))
    x = Conv1D(32, kernel_size=5, activation='relu')(_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    return Model(inputs=_input, outputs=x, name='embedding')


all_features = read.read()
test_ids = list(all_features.keys())

for test_id in test_ids:

    _train_features, _test_features = split(all_features, test_id)

    _train_features, _train_labels = flatten(_train_features)
    _test_features, _test_labels = flatten(_test_features)

    _train_labels_ = np_utils.to_categorical(_train_labels, len(read.activity_list))
    _test_labels_ = np_utils.to_categorical(_test_labels, len(read.activity_list))

    _train_features = np.array(_train_features)
    _train_features = np.expand_dims(_train_features, 3)
    print(_train_features.shape)

    _test_features = np.array(_test_features)
    _test_features = np.expand_dims(_test_features, 3)
    print(_test_features.shape)

    _input_ = Input(shape=(read.dct_length*3*len(read.sensors), 1))
    base_network = conv()
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
    k = 3
    acc = cos_knn(k, _test_preds, _test_labels, _train_preds, _train_labels)

    read.write_data('conv.csv', 'score'+','.join([str(f) for f in results])+'knn_acc'+str(acc))
