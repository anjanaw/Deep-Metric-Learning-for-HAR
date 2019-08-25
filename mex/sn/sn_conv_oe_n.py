import numpy as np
import random
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Conv1D, MaxPooling1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import read
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(2)

num_test_classes = 2
mini_batch_size = 200
batch_size = 60
steps_per_epoch = mini_batch_size
feature_length = read.dct_length * 3 * len(read.sensors)
epochs = 10
k_shot = 5
k = 3


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


def build_conv_model():
    _input = Input(shape=(feature_length, 1))
    x = Conv1D(12, kernel_size=3, activation='relu')(_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1200, activation='relu')(x)
    return Model(inputs=_input, outputs=x, name='embedding')


feature_data = read.read()

test_ids = list(feature_data.keys())
all_labels = list(feature_data[test_ids[0]].keys())

for test_id in test_ids:
    for _int in range(5):
        test_labels_indices = np.random.choice(len(all_labels), num_test_classes, False)
        test_labels = [a for ii, a in enumerate(all_labels) if ii in test_labels_indices]
        print(test_labels)
        train_labels = [a for ii, a in enumerate(all_labels) if ii not in test_labels_indices]
        print(train_labels)
        _train_data, _test_data = read.split(feature_data, test_id)
        _train_data = read.remove_class(_train_data, test_labels)

        _support_data, _test_data = read.support_set_split(_test_data, k_shot)

        _train_data, _train_labels = read.flatten(_train_data)
        _support_data, _support_labels = read.flatten(_support_data)

        _train_data = np.array(_train_data)
        _train_data = np.expand_dims(_train_data, 3)
        _support_data = np.array(_support_data)
        _support_data = np.expand_dims(_support_data, 3)

        _train_labels = np.array(_train_labels)
        _support_labels = np.array(_support_labels)

        base_network = build_conv_model()

        input_a = Input(shape=(feature_length, 1))
        input_b = Input(shape=(feature_length, 1))

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        model = Model(input=[input_a, input_b], output=distance)

        model.compile(loss=contrastive_loss, optimizer='adam')

        for x in range(epochs):
            digit_indices = [np.where(_train_labels == i)[0] for i in train_labels]
            x_pairs, y_pairs = create_pairs(_train_data, digit_indices, len(train_labels))
            model.fit([x_pairs[:, 0], x_pairs[:, 1]], y_pairs,  verbose=1, batch_size=batch_size, epochs=1)

        _support_preds = base_network.predict(_support_data)

        for _l in list(_test_data[test_id].keys()):
            _test_label_data = _test_data[test_id][_l]
            _test_labels = [_l for i in range(len(_test_label_data))]
            _test_label_data = np.array(_test_label_data)
            _test_label_data = np.expand_dims(_test_label_data, 3)
            _test_labels = np.array(_test_labels)
            _test_preds = base_network.predict(_test_label_data)

            acc = read.cos_knn(k, _test_preds, _test_labels, _support_preds, _support_labels)
            result = 'sn_conv, 3nn,' + str(num_test_classes) + ',' + str(test_id) + ',' + ','.join([str(t) for t in test_labels]) + ',' + str(_l) + ',' + str(acc)
            read.write_data('sn_conv_oe_n.csv', result)

