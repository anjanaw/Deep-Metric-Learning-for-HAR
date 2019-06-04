import numpy as np
from scipy import fftpack
import os
import csv
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(2)

window_length = 500
dct_length = 60
data_path = '/Users/anjanawijekoon/Data/SELFBACK/activity_data_34/merge/'
imus = [1, 2]

classes = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
ids = range(len(classes))
classDict = dict(zip(classes, ids))


def read_data(path):
    person_data = {}
    files = os.listdir(path)
    for f in [ff for ff in files if ff != '.DS_Store']:
        temp = f.split("_")
        user = temp[0]
        activity = temp[1]
        data = []
        reader = csv.reader(open(os.path.join(path, f), "r"), delimiter=",")
        for row in reader:
            data.append(row)

        activity_data = {}
        if user in person_data:
            activity_data = person_data[user]
            activity_data[activity] = data
        else:
            activity_data[activity] = data
        person_data[user] = activity_data

    return person_data


def extract_features(data):
    people = {}
    for person in data:
        person_data = data[person]
        classes = {}
        for activity in person_data:
            df = person_data[activity]
            wts = split_windows(df)
            act = classDict[activity]
            dct_wts = dct(wts)
            classes[act] = dct_wts
        people[person] = classes
    return people


def split_windows(data, overlap_ratio=1):
    outputs = []
    i = 0
    N = len(data)
    increment = int(window_length * overlap_ratio)
    while i + window_length < N:
        start = i
        end = start + window_length
        outs = [a[:] for a in data[start:end]]
        i = int(i + increment)
        outputs.append(outs)
    return outputs


def dct(windows):
    dct_window = []
    for tw in windows:
        all_acc_dcts = np.array([])
        for index in imus:
            _index = index - 1
            x = [t[(_index * 3) + 0] for t in tw]
            y = [t[(_index * 3) + 1] for t in tw]
            z = [t[(_index * 3) + 2] for t in tw]

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
    user_data = read_data(data_path)
    feature_data = extract_features(user_data)
    return feature_data

