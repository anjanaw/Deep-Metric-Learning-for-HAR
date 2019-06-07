import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import read
import sys

#k = 3


def cos_knn(k, test_data, test_labels, stored_data, stored_target):
    cosine = cosine_similarity(test_data, stored_data)
    top = [(heapq.nlargest(k, range(len(i)), i.take)) for i in cosine]
    top = [[stored_target[j] for j in i[:k]] for i in top]
    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)
    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == pred[j]:
            correct += 1
    print(correct/float(len(test_labels)))


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

cos_knn(3, _test_data, _test_labels, _train_data, _train_labels)