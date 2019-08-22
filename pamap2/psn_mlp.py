import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras import backend as K
import read
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(2)

mini_batch_size = 200
batch_size = 60
steps_per_epoch = mini_batch_size
feature_length = read.dct_length * 3 * 3
epochs = 20
k = 3
candidates = 5


def get_neighbours(instance, dataset, n):
    return np.argsort(np.linalg.norm(dataset - instance, axis=1))[:n]


def get_accuracy(test_labels, predictions):
    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == predictions[j]:
            correct += 1
    return (correct / float(len(test_labels))) * 100.0


# Define Euclidean distance function
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


# Define the shape of the output of Euclidean distance
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# Define the contrastive loss function (as from Hadsell et al [1].)
def contrastive_loss(y_true, y_pred):
    margin = 15
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def prototype(x, data_samples, no_samples):
    data_samples = random.sample(data_samples, no_samples)
    data_samples = [x[index] for index in data_samples]
    data_samples = np.asarray(data_samples)
    return data_samples.mean(0)


def create_pairs(_x, _digit_indices, num_classes):
    pairs = []
    labels = []
    n = min([len(_digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1 = _digit_indices[d][i],
            z2 = prototype(_x, list(_digit_indices[d]), candidates)
            pairs += [[_x[z1], z2]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = _digit_indices[d][i], prototype(_x, list(_digit_indices[dn]), 5)
            pairs += [[_x[z1], z2]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def build_mlp_model():
    base_input = Input((feature_length,))
    x = Dense(1200, activation='relu')(base_input)
    embedding_model = Model(base_input, x, name='embedding')
    return embedding_model


feature_data = read.read()

test_ids = list(feature_data.keys())
for test_id in test_ids:
    _train_data, _test_data = read.split(feature_data, test_id)
    _train_data, _train_labels = read.flatten(_train_data)
    _test_data, _test_labels = read.flatten(_test_data)

    _train_data = np.array(_train_data)
    _test_data = np.array(_test_data)

    _train_labels = np.array(_train_labels)
    _test_labels = np.array(_test_labels)

    base_network = build_mlp_model()

    input_a = Input(shape=(feature_length,))
    input_b = Input(shape=(feature_length,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    model.compile(loss=contrastive_loss, optimizer='adam')

    for x in range(epochs):
        digit_indices = [np.where(_train_labels == i)[0] for i in range(len(read.classes))]
        x_pairs, y_pairs = create_pairs(_train_data, digit_indices, len(read.classes))
        model.fit([x_pairs[:, 0], x_pairs[:, 1]], y_pairs,  verbose=1, batch_size=batch_size, epochs=1)

    _train_preds = base_network.predict(_train_data)
    _test_preds = base_network.predict(_test_data)

    acc = read.cos_knn(k, _test_preds, _test_labels, _train_preds, _train_labels)
    result = 'prototype_sn_mlp, 3nn,' + str(test_id) + ',' + str(acc)
    print(result)
    read.write_data('sn_mlp.csv', result)
