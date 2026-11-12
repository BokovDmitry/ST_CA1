import pickle
from pathlib import Path

import numpy as np

from .class_names import CLASS_TO_ID

KEEP_LABELS = (
    'cattle', 'fox', 'baby', 'boy', 'girl', 'man', 'woman', 'rabbit',
    'squirrel', 'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
    'lawn_mower', 'tractor',
)
KEEP_SUPERCLASSES = ('trees',)
DATA_DIR = Path(__file__).resolve().parents[2] / 'data/original/cifar-100-python'
META_PATH = DATA_DIR / 'meta'
TRAIN_DATA_PATH = DATA_DIR / 'train'
TEST_DATA_PATH = DATA_DIR / 'test'


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file, encoding='bytes')


def filter_data(data, fine_names, coarse_names):
    fine_labels = data[b'fine_labels']
    coarse_labels = data[b'coarse_labels']
    target_labels = []
    keep_mask = []

    for fine_id, coarse_id in zip(fine_labels, coarse_labels):
        fine_name = fine_names[fine_id]
        coarse_name = coarse_names[coarse_id]
        if fine_name in KEEP_LABELS:
            target_labels.append(CLASS_TO_ID[fine_name])
            keep_mask.append(True)
        elif coarse_name in KEEP_SUPERCLASSES:
            target_labels.append(CLASS_TO_ID[coarse_name])
            keep_mask.append(True)
        else:
            target_labels.append(-1)
            keep_mask.append(False)

    mask = np.asarray(keep_mask)
    filtered = {
        'data': data[b'data'][mask],
        'labels': np.asarray(target_labels)[mask],
        'filenames': [name for name, keep in zip(data[b'filenames'], mask) if keep],
    }
    validate_data(filtered)
    return filtered


def validate_data(data):
    assert len(data['data']) == len(data['labels']) == len(data['filenames']), 'fields out of sync'
    allowed_ids = {CLASS_TO_ID[label] for label in (*KEEP_LABELS, *KEEP_SUPERCLASSES)}
    assert set(np.unique(data['labels'])).issubset(allowed_ids), 'unexpected label or superclass'


def _generate_set(path):
    data = load_pickle(path)
    metadata = load_pickle(META_PATH)
    fine_names = [name.decode('utf-8') for name in metadata[b'fine_label_names']]
    coarse_names = [name.decode('utf-8') for name in metadata[b'coarse_label_names']]
    return filter_data(data, fine_names, coarse_names)


def generate_set():
    return _generate_set(TRAIN_DATA_PATH)


def generate_test_set():
    return _generate_set(TEST_DATA_PATH)
