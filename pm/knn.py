import numpy as np
import read

feature_data = read.read()
test_ids = list(feature_data.keys())
for test_id in test_ids:

    _train_data, _test_data = read.split(feature_data, [test_id])
    _test_data, _test_labels = read.flatten(_test_data)
    _train_data, _train_labels = read.flatten(_train_data)

    _train_data = np.array(_train_data)
    print(_train_data.shape)
    _train_data = np.reshape(_train_data, (_train_data.shape[0], _train_data.shape[1] * _train_data.shape[2]))
    print(_train_data.shape)

    _test_data = np.array(_test_data)
    print(_test_data.shape)
    _test_data = np.reshape(_test_data, (_test_data.shape[0], _test_data.shape[1] * _test_data.shape[2]))
    print(_test_data.shape)

    acc = read.cos_knn(1, _test_data, _test_labels, _train_data, _train_labels)
    print(acc)
    acc = read.cos_knn(2, _test_data, _test_labels, _train_data, _train_labels)
    print(acc)
    acc = read.cos_knn(3, _test_data, _test_labels, _train_data, _train_labels)
    print(acc)
    acc = read.cos_knn(5, _test_data, _test_labels, _train_data, _train_labels)
    print(acc)