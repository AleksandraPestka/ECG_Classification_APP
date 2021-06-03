import os

import pandas as pd
import plotly_express as px
import streamlit as st
from scipy.signal import find_peaks


class PtbXl:
    FREQUENCY = 100
    SAMPLES = 1000
    PATIENTS = 'patient.csv'
    META = 'meta.csv'

    def load_data(self, data_dir, patient):
        df = pd.read_csv(os.path.join(data_dir, self.get_file_name(patient)))
        patient_id = patient[-1]
        patient_data = df[df['ecg_id'] == int(patient_id)]
        return patient_data['channel-0'].values[:self.SAMPLES]

    def show_metadata(self, data_dir, patient):
        df = pd.read_csv(os.path.join(data_dir, self.META))
        patient_id = patient[-1]
        patient_col = df[df['ecg_id'] == int(patient_id)]
        age = patient_col['age']
        sex = 'Man' if patient_col['sex'].any() == 0 else 'Women'
        weight = patient_col['weight']
        st.write(pd.DataFrame({
            'Patient': patient_id,
            'Age': age,
            'Sex': sex,
            'Weight': weight
        }))

    @staticmethod
    def get_file_name(patient):
        return "patient" + patient[-1] + ".csv"

    @staticmethod
    def line_plot(ecg):
        fig = px.line(data_frame=ecg)
        fig.update_layout(
            title='Heart beats [10s]',
            xaxis_title='Samples',
            yaxis_title='Amplitude')
        st.plotly_chart(fig)

    def calculate_heart_rate(self, ecg):
        peaks_all, _ = find_peaks(ecg)
        min_prominence = 0.4 * (max(ecg) - min(ecg))
        min_distance = 50
        r_peaks = find_peaks(ecg, prominence=(min_prominence, None), distance=min_distance)
        rr_distance = []
        rr_time = []
        for i in range(0, len(r_peaks[0]) - 1):
            rr_distance.append(r_peaks[0][i + 1] - r_peaks[0][i])
            rr_time.append(rr_distance[i] / self.FREQUENCY)
        rr_average_time = sum(rr_time) / len(rr_time)
        bpm = 60 / rr_average_time
        return round(bpm, 1)
