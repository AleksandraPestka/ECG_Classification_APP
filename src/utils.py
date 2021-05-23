import os

import pandas as pd
import numpy as np

def load_data(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), header=None)
    test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), header=None)
    return train_df, test_df

def get_sample_signals(data):
    ''' Get 1 sample signal for each class. '''

    X = data.values[:, :-1]
    y = data.values[:, -1].astype(int)
    unique_classes = len(set(y))

    # get class indexes
    sample_data_buffer = []
    for class_no in range(unique_classes):
        sample_data_indexes = np.argwhere(y == class_no).flatten()
        sample_data = X[sample_data_indexes, :][0]
        sample_data_buffer.append(sample_data)
    
    return sample_data_buffer

def prepare_data(data, target=True):
    ''' Prepare data for training/testing. 
    If target is True: split data into input and output. '''

    if target:
        X, y = data.drop([187], axis=1), data[187]
    else:
        X = data
    X = np.array(X).reshape(X.shape[0], X.shape[1], 1)
    if target:
        return X, y
    return X

