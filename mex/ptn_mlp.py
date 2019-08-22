from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K

import random
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import read
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(2)

mini_batch_size = 200
batch_size = 60
steps_per_epoch = mini_batch_size
feature_length = read.dct_length * 3 * len(read.sensors)
epochs = 10
k = 3
candidates = 5


def prototype(x, data_samples, no_samples):
    data_samples = random.sample(data_samples, no_samples)
    data_samples = [x[index] for index in data_samples]
    data_samples = np.asarray(data_samples)
    return data_samples.mean(0)


def get_neighbours(instance, dataset, n):
    return np.argsort(np.linalg.norm(dataset - instance, axis=1))[:n]


def get_triples_minibatch_indices_me(x, dictionary):
    triples_indices = []
    for k in dictionary.keys():
        for value in dictionary[k]:
            anchor = x[value]
            positive = prototype(x, dictionary[k], candidates)
            negative_labels = np.arange(len(read.activity_list))
            negative_label = random.choice(np.delete(negative_labels, np.argwhere(negative_labels==k)))
            negative = prototype(x, dictionary[negative_label], candidates)
            triples_indices.append([anchor, positive, negative])
    random.shuffle(triples_indices)
    return np.asarray(triples_indices)


def get_triples_minibatch_data_u(x, dictionary):
    indices = get_triples_minibatch_indices_me(x, dictionary)
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


feature_data = read.read()

test_ids = list(feature_data.keys())
for test_id in test_ids[0]:

    _train_data, _test_data = read.split(feature_data, test_id)
    _train_data, _train_labels = read.flatten(_train_data)
    _test_data, _test_labels = read.flatten(_test_data)

    _train_data = np.array(_train_data)
    _test_data = np.array(_test_data)

    _embedding_model, _triplet_model = build_mlp_model((feature_length,))

    _triplet_model.fit_generator(triplet_generator_minibatch(_train_data, _train_labels, mini_batch_size),
                                 steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)

    _train_preds = _embedding_model.predict(_train_data)
    _test_preds = _embedding_model.predict(_test_data)

    acc = read.cos_knn(k, _test_preds, _test_labels, _train_preds, _train_labels)
    result = 'prototype_tn_conv, 3nn,' + str(test_id) + ',' + str(acc)
    print(result)
    read.write_data('tn_conv.csv', result)
