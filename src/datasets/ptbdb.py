import os

import pandas as pd


class PtbDb:
    NAME = "PTBDB"
    CLASS_LABELS = ['N: Normal beat', 'A: abnormal']
    NORMAL = "ptbdb_normal.csv"
    ABNORMAL = "ptbdb_abnormal.csv"
    TRAIN = "ptbdb_train.csv"
    TEST = "ptbdb_test.csv"

    def load_data(self, data_dir):
        if not os.path.exists(os.path.join(data_dir, self.TRAIN)) \
            and not os.path.exists(os.path.join(data_dir, self.TEST)):
            self.transform_files(data_dir)
        train_df = pd.read_csv(os.path.join(data_dir, self.TRAIN), header=None)
        test_df = pd.read_csv(os.path.join(data_dir, self.TEST), header=None)
        return train_df, test_df

    def transform_files(self, data_dir):
        ''' Transform files containing normal and abnormal cases 
        to files containing training and test cases. '''

        with open(os.path.join(data_dir, self.NORMAL)) as nf, \
            open(os.path.join(data_dir, self.ABNORMAL)) as af, \
            open(os.path.join(data_dir, self.TRAIN), "a") as trf, \
            open(os.path.join(data_dir, self.TEST), "a") as tef:
            self.split_file(nf, trf, tef)
            self.split_file(af, trf, tef)

    def split_file(self, input_file, train_file, test_file, test_ratio=0.8):
        lines = input_file.read().splitlines()
        border_index = round(test_ratio * len(lines))
        train = lines[:border_index]
        test = lines[border_index:]

        for line in train:
            train_file.write(line + "\n")
        for line in test:
            test_file.write(line + "\n")

