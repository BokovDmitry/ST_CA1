import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

KEEP_LABELS = ['cattle', 'fox', 'baby', 'boy', 'girl', 'man', 'woman', 'rabbit', 'squirrel', 'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'tractor']
KEEP_SUPERCLASS = ['trees']
META_PATH = "data/original/cifar-100-python/meta"
TRAIN_DATA_PATH = "data/original/cifar-100-python/train"
TEST_DATA_PATH = "data/original/cifar-100-python/test"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def generate_labels(path, label_key):
    with open(path, 'rb') as fo:
        labels_dict = pickle.load(fo, encoding='bytes')
    labels = {}
    for i, label in enumerate(labels_dict[label_key]):
        labels[i] = label.decode('utf-8')

    return labels

def filter_data(data, labels_dict, superclass_dict, labels_to_keep, superclass_to_keep):
    keep_labels = [key for key, value in labels_dict.items() if value in labels_to_keep]
    keep_superclass = [key for key, value in superclass_dict.items() if value in superclass_to_keep]
    mask = np.isin(data[b'fine_labels'], keep_labels) | np.isin(data[b'coarse_labels'], keep_superclass)
    return {
        b'data':        data[b'data'][mask],
        b'fine_labels': list(compress(data[b'fine_labels'], mask)),
        b'coarse_labels': list(compress(data[b'coarse_labels'], mask)),
        b'filenames':   list(compress(data[b'filenames'], mask)),
    }

def test_filtered_data(data, labels_dict, superclass_dict, labels_to_keep, superclass_to_keep):
    keep_labels = [key for key, value in labels_dict.items() if value in labels_to_keep]
    keep_superclass = [key for key, value in superclass_dict.items() if value in superclass_to_keep]
    assert(all(fine in keep_labels or coarse in keep_superclass for fine, coarse in zip(data[b'fine_labels'], data[b'coarse_labels']))), "unexpected label or superclass"
    # we have the right number of elements 
    assert(len(data.get(b'data')) == len(data.get(b'fine_labels')) == len(data.get(b'filenames')) == len(data.get(b'coarse_labels'))), "fields out of sync"

def generate_set():
    data = unpickle(TRAIN_DATA_PATH)
    labels = generate_labels(META_PATH, b'fine_label_names')
    superclasses = generate_labels(META_PATH, b'coarse_label_names')
    filtered_data = filter_data(data, labels, superclasses, KEEP_LABELS, KEEP_SUPERCLASS)
    test_filtered_data(filtered_data, labels, superclasses, KEEP_LABELS, KEEP_SUPERCLASS)
    return filtered_data

def generate_test_set():
    data = unpickle(TEST_DATA_PATH)
    labels = generate_labels(META_PATH, b'fine_label_names')
    superclasses = generate_labels(META_PATH, b'coarse_label_names')
    filtered_data = filter_data(data, labels, superclasses, KEEP_LABELS, KEEP_SUPERCLASS)
    test_filtered_data(filtered_data, labels, superclasses, KEEP_LABELS, KEEP_SUPERCLASS)
    return filtered_data
