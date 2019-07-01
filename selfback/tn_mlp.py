from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K

import random
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import sys
import read
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(2)

mini_batch_size = 200
batch_size = 60
steps_per_epoch = mini_batch_size
feature_length = read.dct_length * 3 * len(read.imus)
epochs = 10


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
    read.write_data('tn_mlp.csv', 'tn_mlp,3nn,'+str(test_ids[int(sys.argv[1])])+','+str(correct/float(len(test_labels))))


def get_neighbours(instance, dataset, n):
    return np.argsort(np.linalg.norm(dataset - instance, axis=1))[:n]


def get_triples_minibatch_indices_me(dictionary):
    triples_indices = []
    for k in dictionary.keys():
        for value in dictionary[k]:
            anchor = value
            positive = random.choice(dictionary[k])
            negative_labels = np.arange(len(read.classes))
            negative_label = random.choice(np.delete(negative_labels, np.argwhere(negative_labels == k)))
            negative = random.choice(dictionary[negative_label])
            triples_indices.append([anchor, positive, negative])

    return np.asarray(triples_indices)


def get_triples_minibatch_data_u(x, dictionary):
    indices = get_triples_minibatch_indices_me(dictionary)
    return x[indices[:, 0]], x[indices[:, 1]], x[indices[:, 2]]


def triplet_generator_minibatch(x, y, no_minibatch):
    grouped = defaultdict(list)
    dict_list = []

    for i, label in enumerate(y):
        grouped[label].append(i)

    for k in range(len(grouped)):
        random.shuffle(grouped[k])

    for j in range(no_minibatch):
        dictionary = {}

        for k in range(len(grouped)):
            ran_sam = random.sample(grouped[k], 3)
            dictionary[k] = ran_sam

        dict_list.append(dictionary)

    i = 0

    while 1:
        x_anchor, x_positive, x_negative = get_triples_minibatch_data_u(x, dict_list[i])

        if i == (no_minibatch - 1):
            i = 0
        else:
            i += 1

        yield ({'anchor_input': x_anchor,
                'positive_input': x_positive,
                'negative_input': x_negative},
               None)


def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.mean(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.mean(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)


def build_mlp_model(input_shape):
    base_input = Input(input_shape)
    x = Dense(1200, activation='relu')(base_input)
    embedding_model = Model(base_input, x, name='embedding')

    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]

    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    triplet_model.compile(loss=None, optimizer='adam')

    return embedding_model, triplet_model


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


feature_data = read.read()

test_ids = list(feature_data.keys())
test_id = [test_ids[0]]#[test_ids[sys.argv[1]]]

_train_data, _test_data = split(feature_data, test_id)
_train_data, _train_labels = flatten(_train_data)
_test_data, _test_labels = flatten(_test_data)

_train_data = np.array(_train_data)
_test_data = np.array(_test_data)

_embedding_model, _triplet_model = build_mlp_model((feature_length,))

_triplet_model.fit_generator(triplet_generator_minibatch(_train_data, _train_labels, mini_batch_size),
                             steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)

_train_preds = _embedding_model.predict(_train_data)
_test_preds = _embedding_model.predict(_test_data)

predictions = []
k = 3

cos_knn(k, _test_preds, _test_labels, _train_preds, _train_labels)
