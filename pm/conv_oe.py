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
k_shot = 5
k = 3
feature_length = read.window_length * read.frames_per_second * read.frame_size


def conv():
    _input = Input(shape=(feature_length, 1))
    x = Conv1D(12, kernel_size=5, activation='relu')(_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    return Model(inputs=_input, outputs=x, name='embedding')


all_features = read.read()
test_ids = list(all_features.keys())
all_labels = list(all_features[test_ids[0]].keys())

for test_id in test_ids:
    for a_label in all_labels:
        train_labels = [a for a in all_labels if a != a_label]
        _train_features, _test_features = read.split(all_features, test_id)
        _train_features = read.remove_class(_train_features, [a_label])

        _support_features, _test_features = read.support_set_split(_test_features, k_shot)

        _train_features, _train_labels = read.flatten(_train_features)
        _support_features, _support_labels = read.flatten(_support_features)

        id_list = range(len(train_labels))
        activity_id_dict = dict(zip(train_labels, id_list))

        _train_labels_ = []
        for item in _train_labels:
            _train_labels_.append(activity_id_dict.get(item))

        _train_labels_ = np_utils.to_categorical(_train_labels_, len(train_labels))

        _train_features = np.array(_train_features)
        _train_features = np.reshape(_train_features, (_train_features.shape[0], _train_features.shape[1] * _train_features.shape[2]))
        _train_features = np.expand_dims(_train_features, 3)
        print(_train_features.shape)

        _support_features = np.array(_support_features)
        _support_features = np.reshape(_support_features, (_support_features.shape[0], _support_features.shape[1] * _support_features.shape[2]))
        _support_features = np.expand_dims(_support_features, 3)
        print(_support_features.shape)

        _input_ = Input(shape=(feature_length, 1))
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
            _test_label_data = np.reshape(_test_label_data, (_test_label_data.shape[0], _test_label_data.shape[1] * _test_label_data.shape[2]))
            _test_label_data = np.expand_dims(_test_label_data, 3)
            _test_preds = base_network.predict(_test_label_data)

            acc = read.cos_knn(k, _test_preds, _test_labels, _support_preds, _support_labels)
            result = 'conv, 3nn,' + str(test_id) + ',' + str(a_label) + ',' + str(_l) + ',' + str(acc)
            read.write_data('conv_oe.csv', result)
