import numpy as np
import random
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras import backend as K
import read
import tensorflow as tf
import sys

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
    print(str(correct / float(len(test_labels))))
    read.write_data('sn_mlp.csv',
                    'sn_mlp,3nn,' + str(test_ids[int(sys.argv[1])]) + ',' + str(correct / float(len(test_labels))))


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
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices, num_classes):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def build_mlp_model(input_shape):
    base_input = Input((input_shape,))
    x = Dense(1200, activation='relu')(base_input)
    embedding_model = Model(base_input, x, name='embedding')
    return embedding_model


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


def split(_data, _test_ids):
    train_data_ = {key: value for key, value in _data.items() if key not in _test_ids}
    test_data_ = {key: value for key, value in _data.items() if key in _test_ids}
    return train_data_, test_data_


feature_data = read.read()

test_ids = list(feature_data.keys())
test_id = [test_ids[sys.argv[1]]]

_train_data, _test_data = split(feature_data, test_id)
_train_data, _train_labels = flatten(_train_data)
_test_data, _test_labels = flatten(_test_data)

_train_data = np.array(_train_data)
_test_data = np.array(_test_data)

_train_labels = np.array(_train_labels)
_test_labels = np.array(_test_labels)

base_network = build_mlp_model(feature_length)

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

k = 3
cos_knn(k, _test_preds, _test_labels, _train_preds, _train_labels)

