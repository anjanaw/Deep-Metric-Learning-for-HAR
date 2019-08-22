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
k_shot = 10
k = 3


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
all_labels = list(all_features[test_ids[0]].keys())

for test_id in test_ids:
    for a_label in all_labels:
        train_labels = [a for a in all_labels if a != a_label]
        _train_features, _test_features = split(all_features, test_id)
        _train_features = read.remove_class(_train_features, [a_label])

        _support_features, _test_features = read.support_set_split(_test_features, k_shot)

        _train_features, _train_labels = flatten(_train_features)
        _support_features, _support_labels = flatten(_support_features)

        id_list = range(len(train_labels))
        activity_id_dict = dict(zip(train_labels, id_list))

        _train_labels_ = []
        for item in _train_labels:
            _train_labels_.append(activity_id_dict.get(item))

        _train_labels_ = np_utils.to_categorical(_train_labels_, len(train_labels))

        _train_features = np.array(_train_features)
        _train_features = np.expand_dims(_train_features, 3)
        print(_train_features.shape)

        _support_features = np.array(_support_features)
        _support_features = np.expand_dims(_support_features, 3)
        print(_support_features.shape)

        _input_ = Input(shape=(read.dct_length*3*len(read.sensors), 1))
        base_network = conv()
        base = base_network(_input_)
        classifier = Dense(len(train_labels), activation='softmax')(base)
        _model = Model(inputs=_input_, outputs=classifier, name='classifier')

        _model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        _model.fit(_train_features, _train_labels_, verbose=1, batch_size=batch_size, epochs=epochs, shuffle=True)

        _support_preds = base_network.predict(_support_features)

        # knn evaluation
        for _l in list(_test_features[test_id].keys()):
            _test_label_data = _test_features[test_id][_l]
            _test_labels = [_l for i in range(len(_test_label_data))]
            _test_label_data = np.array(_test_label_data)
            _test_label_data = np.expand_dims(_test_label_data, 3)
            _test_preds = base_network.predict(_test_label_data)

            acc = cos_knn(k, _test_preds, _test_labels, _support_preds, _support_labels)
            result = 'conv, 3nn,' + str(test_id) + ',' + str(a_label) + ',' + str(_l) + ',' + str(acc)
            read.write_data('conv_oe.csv', result)
