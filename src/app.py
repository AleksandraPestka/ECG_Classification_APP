import numpy as np
import pandas as pd
import streamlit as st
import plotly_express as px

from utils import load_data, get_sample_signals
from classification import run_testing
from config import Config

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

def upload_new_data():
    uploaded_file = st.file_uploader("Upload CSV file with new signals", type=['.csv'])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file, header=None)
        return new_data

if __name__ == '__main__':
    st.title('ECG Classification App')
    st.header('Dataset visualization')

    CLASS_LABELS = ['N: Normal beat',
                    'S: Supraventricular ectopic beats',
                    'V: Ventricular ectopic beats ',
                    'F: Fusion Beats',
                    'Q: Unknown Beats']

    train, test = load_data(Config.data_dir)
    plot_class_percentage(train, CLASS_LABELS)
    plot_example_ecg(train, CLASS_LABELS)

    st.header('Testing')
    new_data = upload_new_data()
    if new_data is not None:
        predictions, loss, acc_score = run_testing(new_data, Config.model_dir)
        st.write(f'Accuracy score: {acc_score:.2f}')
        st.write(f'Sparse categorical crossentropy loss: {loss:.2f}')

    