import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

KEEP_LABELS = ['automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck']
LABELS_PATH = "data/original/cifar-10-batches-py/batches.meta"
GENERAL_BATCH_PATH = "data/original/cifar-10-batches-py/data_batch_"
NUMBER_OF_BATCHES = 5

def unpickle(file, number_of_batches):
    batches = []
    for i in range(number_of_batches):
        with open(file+str(i+1), 'rb') as fo:
            batches.append(pickle.load(fo, encoding='bytes'))
    return {
        b'batch_label': batches[0][b'batch_label'],
        b'data': np.concatenate([batch[b'data'] for batch in batches]),
        b'labels': [label for batch in batches for label in batch[b'labels']],
        b'filenames': [filename for batch in batches for filename in batch[b'filenames']],
    }

def generate_labels(path):
    with open(path, 'rb') as fo:
        labels_dict = pickle.load(fo, encoding='bytes')
    labels = {}
    for i in range(len(labels_dict.get(b'label_names'))):
        labels[i] = labels_dict.get(b'label_names')[i].decode('utf-8')

    print(labels)
    return labels

# Claude helped to find the most efficient way to filter out the redundant labels 
def filter_data(data, labels_dict, labels_to_keep):
    keep = [key for key, value in labels_dict.items() if value in labels_to_keep]
    print(f"labels_dict: {labels_dict} \nResulting labels: {keep}")
    mask = np.isin(data[b'labels'], keep)
    return {
        b'batch_label': data[b'batch_label'],
        b'data':        data[b'data'][mask],
        b'labels':      list(compress(data[b'labels'], mask)),
        b'filenames':   list(compress(data[b'filenames'], mask)),
    }

def test_filtered_data(data, labels_dict, labels_to_keep):
    keep = [key for key, value in labels_dict.items() if value in labels_to_keep]
    # we stored only the required labels
    assert(set(data.get(b'labels')) == set(keep)), "incorrect set of labels"
    # we have the right number of elements 
    assert(len(data.get(b'data')) == len(data.get(b'labels')) == len(data.get(b'filenames'))), "fields out of sync"

labels_path = "data/original/cifar-10-batches-py/batches.meta"
general_batch_path = GENERAL_BATCH_PATH

LABELS = generate_labels(labels_path)

original_data = unpickle(general_batch_path, NUMBER_OF_BATCHES)
filtered_data = filter_data(original_data, LABELS, KEEP_LABELS)
test_filtered_data(filtered_data, LABELS, KEEP_LABELS)
