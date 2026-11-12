import pickle
from pathlib import Path

import numpy as np

from .class_names import CLASS_TO_ID

KEEP_LABELS = ('automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'truck')
DATA_DIR = Path(__file__).resolve().parents[2] / 'data/original/cifar-10-batches-py'
LABELS_PATH = DATA_DIR / 'batches.meta'
TRAIN_BATCH_PREFIX = DATA_DIR / 'data_batch_'
TEST_DATA_PATH = DATA_DIR / 'test_batch'
NUMBER_OF_TRAIN_BATCHES = 5


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file, encoding='bytes')


def load_batches(prefix, count):
    batches = [load_pickle(f'{prefix}{index}') for index in range(1, count + 1)]
    return {
        b'data': np.concatenate([batch[b'data'] for batch in batches]),
        b'labels': [label for batch in batches for label in batch[b'labels']],
        b'filenames': [name for batch in batches for name in batch[b'filenames']],
    }


def filter_data(data, label_names):
    original_labels = data[b'labels']
    target_labels = [CLASS_TO_ID[label_names[label]] for label in original_labels]
    mask = np.isin(target_labels, [CLASS_TO_ID[label] for label in KEEP_LABELS])
    filtered = {
        'data': data[b'data'][mask],
        'labels': np.asarray(target_labels)[mask],
        'filenames': [name for name, keep in zip(data[b'filenames'], mask) if keep],
    }
    validate_data(filtered, KEEP_LABELS)
    return filtered


def validate_data(data, labels_to_keep):
    assert len(data['data']) == len(data['labels']) == len(data['filenames']), 'fields out of sync'
    allowed_ids = {CLASS_TO_ID[label] for label in labels_to_keep}
    assert set(np.unique(data['labels'])).issubset(allowed_ids), 'unexpected label'


def _generate_set(data):
    metadata = load_pickle(LABELS_PATH)
    label_names = [name.decode('utf-8') for name in metadata[b'label_names']]
    return filter_data(data, label_names)


def generate_set():
    data = load_batches(TRAIN_BATCH_PREFIX, NUMBER_OF_TRAIN_BATCHES)
    return _generate_set(data)


def generate_test_set():
    return _generate_set(load_pickle(TEST_DATA_PATH))
