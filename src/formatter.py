import numpy as np
from tensorflow.keras.utils import to_categorical

def merge_sets(set1, set2):
    return {
    'data': np.concatenate([set1['data'], set2['data']]),
    'labels': np.concatenate([set1['labels'], set2['labels']]),
    'filenames': set1['filenames'] + set2['filenames'],
}

def format_images(data):
    data['data'] = data['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
    return data

def format_sets(set1, set2):
    merged_set = merge_sets(set1,set2)
    formatted_set = format_images(merged_set)
    return formatted_set

def one_hot_encode(num_classes, y_train, y_val, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return y_train, y_val, y_test
