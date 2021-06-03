''' Script for training and testing classification model. '''

import os

import pandas as pd
import numpy as np
from keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Dense
from keras.models import Sequential, load_model
from keras.metrics import AUC
from keras.utils import to_categorical

from config import Config
from utils import prepare_data


def balance_data(df, data_name):
    ''' Reduce the number of predominant class instances. '''

    if data_name == 'MITBIH':
        class_1 = df[df.iloc[:, -1] == 1.0]
        class_2 = df[df.iloc[:, -1] == 2.0]
        class_3 = df[df.iloc[:, -1] == 3.0]
        class_4 = df[df.iloc[:, -1] == 4.0]
        class_0 = df[df.iloc[:, -1] == 0.0].sample(n=8000)
        new_df = pd.concat([class_0, class_1, class_2, class_3, class_4])
    elif data_name == 'PTBDB':
        class_0 = df[df.iloc[:, -1] == 0.0]
        class_1 = df[df.iloc[:, -1] == 1.0].sample(n=3300)
        new_df = pd.concat([class_0, class_1])
    return new_df

def create_model(n_input, n_output):
    if n_output == 2:
        # binary classification
        output_units = 1
        output_activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        # multiclass classification
        output_units = n_output
        output_activation = 'softmax'
        loss = 'sparse_categorical_crossentropy' # no need to use OHE

    clf = Sequential()

    clf.add(Conv1D(filters=32, kernel_size=(3,), padding='same',
                   activation='relu', input_shape=(n_input, 1)))
    clf.add(Conv1D(filters=64, kernel_size=(3,), padding='same',
                   activation='relu'))
    clf.add(Conv1D(filters=128, kernel_size=(5,), padding='same',
                   activation='relu'))

    clf.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
    clf.add(Dropout(0.5))

    clf.add(Flatten())

    clf.add(Dense(units=264, activation='relu'))
    clf.add(Dense(units=512, activation='relu'))

    clf.add(Dense(units=output_units, activation=output_activation))
    clf.compile(optimizer='adam',
                loss=loss,
                metrics=['accuracy'])  # TODO: add AUC metric

    return clf


def run_training(train, model_dir, dataset, summary=False):
    train = balance_data(train, dataset.NAME)
    X_train, y_train = prepare_data(train)

    n_input = X_train.shape[1]
    n_output = len(np.unique(y_train))

    clf = create_model(n_input, n_output)
    if summary: clf.summary()
    clf.fit(X_train, y_train, validation_split=0.1, epochs=10, shuffle=True)
    clf.save(os.path.join(model_dir, dataset.NAME))


def run_testing(test, model_dir, dataset_name):
    X_test, y_test = prepare_data(test)
    clf = load_model(os.path.join(model_dir, dataset_name))
    loss, acc = clf.evaluate(X_test, y_test)
    y_pred = clf.predict_classes(X_test)
    return y_pred, loss, acc

def run(dataset):
    train, test = dataset.load_data(Config.data_dir)
    run_training(train, Config.model_dir, dataset)