import os
import pandas as pd


class MitBih:
    NAME = "MITBIH"
    CLASS_LABELS = ['N: Normal beat',
                    'S: Supraventricular ectopic beats',
                    'V: Ventricular ectopic beats ',
                    'F: Fusion Beats',
                    'Q: Unknown Beats']
    TRAIN = 'mitbih_train.csv'
    TEST = 'mitbih_test.csv'

    def load_data(self, data_dir):
        data_dir = os.path.join('..', data_dir)
        train_df = pd.read_csv(os.path.join(data_dir, self.TRAIN), header=None)
        test_df = pd.read_csv(os.path.join(data_dir, self.TEST), header=None)
        return train_df, test_df
