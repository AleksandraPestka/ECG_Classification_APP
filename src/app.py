import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly_express as px

def load_data(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), header=None)
    test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), header=None)
    return train_df, test_df

def get_sample_signals(data):
    ''' Get 1 sample signal for each class. '''

    X = data.values[:, :-1]
    y = data.values[:, -1].astype(int)

    # get class indexes
    sample_data_buffer = []
    for class_no in range(4):
        sample_data_indexes = np.argwhere(y == class_no).flatten()
        sample_data = X[sample_data_indexes, :][0]
        sample_data_buffer.append(sample_data)
    
    return sample_data_buffer

def plot_class_percentage(data, class_labels):
    data_class = data.iloc[:, -1].astype(int)
    counts = data_class.value_counts().rename('counts').sort_index()
    fig = px.pie(counts, values=counts, names=class_labels, title='Class imbalance')
    st.plotly_chart(fig)

def plot_example_ecg(data, class_labels):
    sample_signals = get_sample_signals(data)
    x = np.arange(0, data.shape[1]-1) * 8/1000

    df = pd.DataFrame()
    df['x'] = x
    for label, signal in zip(class_labels, sample_signals):
        df[label] = signal

    selected_class = st.multiselect('Select signal class', class_labels,
                                     default='N: Normal beat')
    fig = px.line(data_frame=df, x='x', y=selected_class)

    fig.update_layout(
        title='1-beat ECG for every category',
        xaxis_title='Time (ms)',
        yaxis_title='Amplitude',
        legend_title_text='Classes')
    st.plotly_chart(fig)

if __name__ == '__main__':
    st.title('ECG Classification App')

    DATA_DIR = '../data'
    CLASS_LABELS = ['N: Normal beat',
                    'S: Supraventricular ectopic beats',
                    'V: Ventricular ectopic beats ',
                    'F: Fusion Beats',
                    'Q: Unknown Beats']

    train, test = load_data(DATA_DIR)
    plot_class_percentage(train, CLASS_LABELS)
    plot_example_ecg(train, CLASS_LABELS)