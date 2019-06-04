import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Lambda, Dense
from keras.layers.merge import _Merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import read
import sys

np.random.seed(1)
tf.set_random_seed(2)

batch_size = 120
samples_per_class = 5
classes_per_set = 9
feature_length = read.dct_length * 3 * len(read.imus)
train_size = 500
epochs = 10

class MatchCosine(_Merge):
    def __init__(self, nway=5, n_samp=1, **kwargs):
        super(MatchCosine, self).__init__(**kwargs)
        self.eps = 1e-10
        self.nway = nway
        self.n_samp = n_samp

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != self.nway * self.n_samp + 2:
            raise ValueError(
                'A ModelCosine layer should be called on a list of inputs of length %d' % (self.nway * self.n_samp + 2))

    def call(self, inputs):
        self.nway = (len(inputs) - 2) / self.n_samp
        similarities = []

        targetembedding = inputs[-2]
        numsupportset = len(inputs) - 2
        for ii in range(numsupportset):
            supportembedding = inputs[ii]

            sum_support = tf.reduce_sum(tf.square(supportembedding), 1, keep_dims=True)
            supportmagnitude = tf.rsqrt(tf.clip_by_value(sum_support, self.eps, float("inf")))

            sum_query = tf.reduce_sum(tf.square(targetembedding), 1, keep_dims=True)
            querymagnitude = tf.rsqrt(tf.clip_by_value(sum_query, self.eps, float("inf")))

            dot_product = tf.matmul(tf.expand_dims(targetembedding, 1), tf.expand_dims(supportembedding, 2))
            dot_product = tf.squeeze(dot_product, [1])

            cosine_similarity = dot_product * supportmagnitude * querymagnitude
            similarities.append(cosine_similarity)

        similarities = tf.concat(axis=1, values=similarities)
        softmax_similarities = tf.nn.softmax(similarities)
        preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities, 1), inputs[-1]))

        preds.set_shape((inputs[0].shape[0], self.nway))
        return preds

    def compute_output_shape(self, input_shape):
        input_shapes = input_shape
        return (input_shapes[0][0], self.nway)


def support_set_split(_data):
    support_set = {}
    everything_else = {}
    for user, labels in _data.items():
        _support_set = {}
        _everything_else = {}
        for label, data in labels.items():
            supportset_indexes = np.random.choice(range(len(data)), samples_per_class, False)
            supportset = [d for index, d in enumerate(data) if index in supportset_indexes]
            everythingelse = [d for index, d in enumerate(data) if index not in supportset_indexes]
            _support_set[label] = supportset
            _everything_else[label] = everythingelse
        support_set[user] = _support_set
        everything_else[user] = _everything_else
    return support_set, everything_else


def packslice(data_set):
    n_samples = samples_per_class * classes_per_set
    support_cacheX = []
    support_cacheY = []
    target_cacheY = []

    for iiii in range(train_size):
        slice_x = np.zeros((n_samples + 1, feature_length))
        slice_y = np.zeros((n_samples,))

        ind = 0
        pinds = np.random.permutation(n_samples)
        classes = np.random.choice(list(data_set.keys()), classes_per_set, False)

        x_hat_class = np.random.randint(classes_per_set)

        for j, cur_class in enumerate(classes):
            data_pack = data_set[cur_class]
            example_inds = np.random.choice(len(data_pack), samples_per_class, False)

            for eind in example_inds:
                slice_x[pinds[ind], :] = data_pack[eind]
                slice_y[pinds[ind]] = cur_class
                ind += 1

            if j == x_hat_class:
                target_indx = np.random.choice(len(data_pack))
                while target_indx in example_inds:
                    target_indx = np.random.choice(len(data_pack))
                slice_x[n_samples, :] = data_pack[target_indx]
                target_y = cur_class

        support_cacheX.append(slice_x)
        support_cacheY.append(keras.utils.to_categorical(slice_y, classes_per_set))
        target_cacheY.append(keras.utils.to_categorical(target_y, classes_per_set))

    return np.array(support_cacheX), np.array(support_cacheY), np.array(target_cacheY)


def create_train_instances(train_sets):
    support_X = None
    support_y = None
    target_y = None

    for user_id, train_feats in train_sets.items():
        _support_X, _support_y, _target_y = packslice(train_feats)

        if support_X is not None:
            support_X = np.concatenate((support_X, _support_X))
            support_y = np.concatenate((support_y, _support_y))
            target_y = np.concatenate((target_y, _target_y))
        else:
            support_X = _support_X
            support_y = _support_y
            target_y = _target_y

    print("Data shapes: ")
    print(support_X.shape)
    print(support_y.shape)
    print(target_y.shape)
    return [support_X, support_y, target_y]


def packslice_test(data_set, support_set):
    n_samples = samples_per_class * classes_per_set
    support_cacheX = []
    support_cacheY = []
    target_cacheY = []

    support_X = np.zeros((n_samples, feature_length))
    support_y = np.zeros((n_samples,))
    for i, _class in enumerate(support_set.keys()):
        X = support_set[_class]
        for j in range(len(X)):
            support_X[(i * samples_per_class) + j, :] = X[j]
            support_y[(i * samples_per_class) + j] = _class

    for _class in data_set:
        X = data_set[_class]
        for iiii in range(len(X)):
            slice_x = np.zeros((n_samples + 1, feature_length))
            slice_y = np.zeros((n_samples,))

            slice_x[:n_samples, :] = support_X[:]
            slice_x[n_samples, :] = X[iiii]

            slice_y[:n_samples] = support_y[:]

            target_y = _class

            support_cacheX.append(slice_x)
            support_cacheY.append(keras.utils.to_categorical(slice_y, classes_per_set))
            target_cacheY.append(keras.utils.to_categorical(target_y, classes_per_set))

    return np.array(support_cacheX), np.array(support_cacheY), np.array(target_cacheY)


def create_test_instance(test_set, support_set):
    support_X = None
    support_y = None
    target_y = None

    for user_id, test_data in test_set.items():
        support_data = support_set[user_id]
        _support_X, _support_y, _target_y = packslice_test(test_data, support_data)

        if support_X is not None:
            support_X = np.concatenate((support_X, _support_X))
            support_y = np.concatenate((support_y, _support_y))
            target_y = np.concatenate((target_y, _target_y))
        else:
            support_X = _support_X
            support_y = _support_y
            target_y = _target_y

    print("Data shapes: ")
    print(support_X.shape)
    print(support_y.shape)
    print(target_y.shape)
    return [support_X, support_y, target_y]


def split(_data, test_ids):
    train_data = {key: value for key, value in _data.items() if key not in test_ids}
    test_data = {key: value for key, value in _data.items() if key in test_ids}
    return train_data, test_data


def get_hold_out_users(users):
    indices = np.random.choice(len(users), int(len(users) / 3), False)
    test_users = [u for indd, u in enumerate(users) if indd in indices]
    return test_users


def mlp_embedding(x):
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    return x


feature_data = read.read()

test_ids = list(feature_data.keys())
test_id = [test_ids[0]]#[sys.argv[1]]]

_train_data, _test_data = split(feature_data, test_id)
train_data = create_train_instances(_train_data)

test_support_set, _test_data = support_set_split(_test_data)
test_data = create_test_instance(_test_data, test_support_set)

model = None
y_pred_p = None
y_true_p = None
numsupportset = samples_per_class * classes_per_set
input1 = Input((numsupportset + 1, feature_length))

modelinputs = []
for lidx in range(numsupportset):
    modelinputs.append(mlp_embedding(Lambda(lambda x: x[:, lidx, :])(input1)))
targetembedding = mlp_embedding(Lambda(lambda x: x[:, -1, :])(input1))
modelinputs.append(targetembedding)
supportlabels = Input((numsupportset, classes_per_set))
modelinputs.append(supportlabels)

knnsimilarity = MatchCosine(nway=classes_per_set, n_samp=samples_per_class)(modelinputs)

model = Model(inputs=[input1, supportlabels], outputs=knnsimilarity)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([train_data[0], train_data[1]], train_data[2], epochs=epochs, batch_size=batch_size, verbose=1)
score = model.evaluate([test_data[0], test_data[1]], test_data[2], batch_size=batch_size, verbose=1)
print(score)
