import numpy as np
from scipy import fftpack
import os
import csv
import tensorflow as tf

tf.set_random_seed(2)
np.random.seed(1)

samples_per_class = 5
train_size = 500
epochs = 20
all_classes = 8
dct_length = 60
window_length = 500
increment_ratio = 1

classes = ["1", "2", "3", "4", "12", "13", "16", "17"]
ids = range(len(classes))
classDict = dict(zip(classes, ids))

optional_path = '/Users/anjanawijekoon/Data/PAMAP2/data/Optional/'
protocol_path = '/Users/anjanawijekoon/Data/PAMAP2/data/Protocol/'
results_path = 'results.csv'


def write_data(data):
    if os.path.isfile(results_path):
        f = open(results_path, 'a')
        f.write(data + '\n')
    else:
        f = open(results_path, 'w')
        f.write(data + '\n')
    f.close()


def read_data():
    data_path = optional_path
    person_data = {}
    person_data = _read(data_path, person_data)
    data_path = protocol_path
    person_data = _read(data_path, person_data)
    return person_data


def _read(data_path, person_data):
    files = os.listdir(data_path)
    for f in files:
        temp = f.replace('subject', '').replace('.dat', '')
        user = temp
        reader = csv.reader(open(os.path.join(data_path, f), "r"), delimiter=" ")
        for row in reader:
            if 'NaN' not in row[4:7] and 'NaN' not in row[21:24] and 'NaN' not in row[38:41]:
                instance = []
                instance.extend([float(t) for t in row[4:7]])
                instance.extend([float(t) for t in row[21:24]])
                instance.extend([float(t) for t in row[38:41]])
                activity = row[1]
                if user in person_data:
                    activity_data = person_data[user]
                    if activity in activity_data:
                        temp = activity_data[activity]
                        temp.append(instance)
                        activity_data[activity] = temp
                    else:
                        temp = []
                        temp.append(instance)
                        activity_data[activity] = temp
                    person_data[user] = activity_data
                else:
                    temp = []
                    temp.append(instance)
                    activity_data = {}
                    activity_data[activity] = temp
                    person_data[user] = activity_data
    return person_data


def keep_class(_data):
    data = {}
    for user_id, labels in _data.items():
        _labels = {}
        for label in labels:
            if label in classes:
                _labels[label] = labels[label]
        data[user_id] = _labels
    return data


def extract_features(person_data):
    people = {}
    for person in person_data:
        p_data = person_data[person]
        classes = {}
        for activity in p_data:
            df = p_data[activity]
            all = split_windows(df)
            dct_all = dct(all)
            act = classDict[activity]
            classes[act] = dct_all
        people[person] = classes
    return people


def split_windows(data):
    modal1 = []
    i = 0
    N = len(data)
    increment = int(window_length * increment_ratio)
    while i + window_length < N:
        start = i
        end = start + window_length
        modal_1 = [a[:] for a in data[start:end]]
        i = int(i + increment)
        modal1.append(modal_1)
    return modal1


def dct(windows):
    dct_window = []
    for tw in windows:
        all_acc_dcts = np.array([])
        for i in range(3):
            x = [t[i * 3 + 0] for t in tw]
            y = [t[i * 3 + 1] for t in tw]
            z = [t[i * 3 + 2] for t in tw]

            dct_x = np.abs(fftpack.dct(x, norm='ortho'))
            dct_y = np.abs(fftpack.dct(y, norm='ortho'))
            dct_z = np.abs(fftpack.dct(z, norm='ortho'))

            v = np.array([])
            v = np.concatenate((v, dct_x[:dct_length]))
            v = np.concatenate((v, dct_y[:dct_length]))
            v = np.concatenate((v, dct_z[:dct_length]))
            all_acc_dcts = np.concatenate((all_acc_dcts, v))
        dct_window.append(all_acc_dcts)
    return dct_window


def read():
    user_data = read_data()
    user_data = keep_class(user_data)
    feature_data = extract_features(user_data)
    return feature_data
