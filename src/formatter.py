import numpy as np

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
