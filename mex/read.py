import csv
import os
import datetime as dt
from scipy import fftpack
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq

frames_per_second = 1
window_length = 5
increment = 2

ac_min_length = 95*window_length
ac_max_length = 100*window_length

path = '/Users/anjanawijekoon/Data/MEx/min'

frame_size = 3
dct_length = 60

sensors = ['acw', 'act']

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))


def cos_knn(k, test_data, test_labels, train_data, train_labels):
    cosim = cosine_similarity(test_data, train_data)

    top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]
    top = [[train_labels[j] for j in i[:k]] for i in top]

    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)

    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == pred[j]:
            correct += 1
    acc = correct / float(len(test_labels))
    return acc


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


def write_data(results_file, data):
    if os.path.isfile(results_file):
        f = open(results_file, 'a')
        f.write(data + '\n')
    else:
        f = open(results_file, 'w')
        f.write(data + '\n')
    f.close()


def remove_class(_data, remove_classes):
    data = {}
    for user_id, labels in _data.items():
        _labels = {}
        for label in labels:
            if label not in remove_classes:
                _labels[label] = labels[label]
        data[user_id] = _labels
    return data


def support_set_split(_data, k_shot):
    support_set = {}
    everything_else = {}
    for user, labels in _data.items():
        _support_set = {}
        _everything_else = {}
        for label, data in labels.items():
            supportset_indexes = np.random.choice(range(len(data)), k_shot, False)
            supportset = [d for index, d in enumerate(data) if index in supportset_indexes]
            everythingelse = [d for index, d in enumerate(data) if index not in supportset_indexes]
            _support_set[label] = supportset
            _everything_else[label] = everythingelse
        support_set[user] = _support_set
        everything_else[user] = _everything_else
    return support_set, everything_else


def _read(_file, _length):
    reader = csv.reader(open(_file, "r"), delimiter=",")
    _data = []
    for row in reader:
        if len(row) == _length:
            if len(row[0]) == 19 and '.' not in row[0]:
                row[0] = row[0]+'.000000'
            temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
            _temp = [float(f) for f in row[1:]]
            temp.extend(_temp)
            _data.append(temp)
    return _data


def _read_():
    alldata = {}
    sensor_path = os.path.join(path, 'acw')
    subjects = os.listdir(sensor_path)
    for subject in subjects:
        allactivities = {}
        subject_path = os.path.join(sensor_path, subject)
        activities = os.listdir(subject_path)
        for activity in activities:
            sensor = activity.split('.')[0].split('_')[1]+'_'+activity.split('.')[0].split('_')[2]
            activity_id = activity.split('.')[0].split('_')[0]
            _data = _read(os.path.join(subject_path, activity), 4)
            if activity_id in allactivities:
                allactivities[activity_id][sensor] = _data
            else:
                allactivities[activity_id] = {}
                allactivities[activity_id][sensor] = _data
        alldata[subject] = allactivities
    sensor_path = os.path.join(path, 'act')
    subjects = os.listdir(sensor_path)
    for subject in subjects:
        allactivities = alldata[subject]
        subject_path = os.path.join(sensor_path, subject)
        activities = os.listdir(subject_path)
        for activity in activities:
            sensor = activity.split('.')[0].split('_')[1]+'_'+activity.split('.')[0].split('_')[2]
            activity_id = activity.split('.')[0].split('_')[0]
            _data = _read(os.path.join(subject_path, activity), 4)
            if activity_id in allactivities:
                allactivities[activity_id][sensor] = _data
            else:
                allactivities[activity_id] = {}
                allactivities[activity_id][sensor] = _data
        alldata[subject] = allactivities
    return alldata


def pad(data, length):
    pad_length = []
    if length % 2 == 0:
        pad_length = [int(length / 2), int(length / 2)]
    else:
        pad_length = [int(length / 2) + 1, int(length / 2)]
    new_data = []
    for index in range(pad_length[0]):
        new_data.append(data[0])
    new_data.extend(data)
    for index in range(pad_length[1]):
        new_data.append(data[len(data) - 1])
    return new_data


def reduce(data, length):
    red_length = []
    if length % 2 == 0:
        red_length = [int(length / 2), int(length / 2)]
    else:
        red_length = [int(length / 2) + 1, int(length / 2)]
    new_data = data[red_length[0]:len(data) - red_length[1]]
    return new_data


def pad_features(_features):
    new_features = {}
    for subject in _features:
        new_activities = {}
        activities = _features[subject]
        for act in activities:
            items = activities[act]
            new_items = []
            for item in items:
                _len = len(item)
                if _len < ac_min_length:
                    continue
                elif _len > ac_max_length:
                    item = reduce(item, _len - ac_max_length)
                    new_items.append(item)
                elif _len < ac_max_length:
                    item = pad(item, ac_max_length - _len)
                    new_items.append(item)
            new_activities[act] = new_items
        new_features[subject] = new_activities
    return new_features


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]


def trim(_data):
    _length = len(_data)
    _inc = _length/(window_length*frames_per_second)
    _new_data = []
    for i in range(window_length*frames_per_second):
        _new_data.append(_data[i*_inc])
    return _new_data


def frame_reduce(_data):
    if frames_per_second == 0:
        return _data
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            time_windows = []
            for item in activity_data:
                time_windows.append(trim(item))
            _activities[activity] = time_windows
        _features[subject] = _activities
    return _features


def split_windows(data):
    outputs = []
    start = data[0][0]
    end = data[len(data) - 1][0]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window_length)

    frames = [a[1:] for a in data[:]]
    frames = np.array(frames)
    _length = frames.shape[0]
    frames = np.reshape(frames, (_length*frame_size))
    if max(frames) != 0:
        frames = frames/max(frames)
    frames = [float("{0:.5f}".format(f)) for f in frames.tolist()]
    frames = np.reshape(np.array(frames), (_length, frame_size))

    while start + _window < end:
        _end = start + _window
        start_index = find_index(data, start)
        end_index = find_index(data, _end)
        instances = [a[:] for a in frames[start_index:end_index]]
        start = start + _increment
        outputs.append(instances)
    return outputs


def dct(windows, sensor, activity, subject):
    dct_window = []
    for tw in windows:
        x = [t[0] for t in tw]
        y = [t[1] for t in tw]
        z = [t[2] for t in tw]
        '''
        for d in x:
            if math.isnan(d):
                print('x:'+str(sensor)+','+str(activity)+','+str(subject))

        for d in y:
            if math.isnan(d):
                print('y:'+str(sensor)+','+str(activity)+','+str(subject))

        for d in z:
            if math.isnan(d):
                print('z:'+str(sensor)+','+str(activity)+','+str(subject))

        print('dct')
        '''
        dct_x = np.abs(fftpack.dct(x, norm='ortho'))
        dct_y = np.abs(fftpack.dct(y, norm='ortho'))
        dct_z = np.abs(fftpack.dct(z, norm='ortho'))
        '''
        for d in dct_x[:dct_length]:
            if math.isnan(d):
                print('x:'+str(sensor)+','+str(activity)+','+str(subject))

        for d in dct_y[:dct_length]:
            if math.isnan(d):
                print('y:'+str(sensor)+','+str(activity)+','+str(subject))

        for d in dct_z[:dct_length]:
            if math.isnan(d):
                print('z:'+str(sensor)+','+str(activity)+','+str(subject))
        '''
        v = np.array([])
        v = np.concatenate((v, dct_x[:dct_length]))
        v = np.concatenate((v, dct_y[:dct_length]))
        v = np.concatenate((v, dct_z[:dct_length]))

        dct_window.append(v)
    return dct_window


def join(acw, act):
    _all = []
    for w, t in zip(acw, act):
        _all.append(np.append(w, t))
    return _all


def extract_features(_data):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            time_windows = {}
            activity_id = activity_id_dict.get(activity)
            activity_data = activities[activity]
            sensors = []
            for sensor in activity_data:
                if sensor in time_windows:
                    time_windows[sensor].extend(split_windows(activity_data[sensor]))
                else:
                    time_windows[sensor] = split_windows(activity_data[sensor])
            _activities[activity_id] = join(dct(time_windows[activity_data.keys()[0]], sensor, activity, subject),
                                            dct(time_windows[activity_data.keys()[1]], sensor, activity, subject))
        _features[subject] = _activities
    return _features


def read():
    all_data = _read_()
    return extract_features(all_data)


def get_candidates(_data, candidates, samples_per_class):
    prototype_candidates = np.random.choice(len(_data), samples_per_class*candidates, False)
    prototypes = []
    for i in prototype_candidates:
        prototypes.append(_data[i])

    examples = []
    for i in range(samples_per_class):
        examples.append(np.mean(prototypes[i*candidates:(i+1)*candidates], axis=0))

    return examples, prototype_candidates
