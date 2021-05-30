''' Script for training and testing classification model. '''

import pandas as pd
from keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Dense
from keras.models import Sequential, load_model

from config import Config
from utils import prepare_data


def balance_data(df):
    ''' Reduce the number of normal class instances. '''

    class_1 = df[df.iloc[:, -1] == 1.0]
    class_2 = df[df.iloc[:, -1] == 2.0]
    class_3 = df[df.iloc[:, -1] == 3.0]
    class_4 = df[df.iloc[:, -1] == 4.0]
    class_0 = df[df.iloc[:, -1] == 0.0].sample(n=8000)
    new_df = pd.concat([class_0, class_1, class_2, class_3, class_4])
    return new_df


def create_model(n_input, n_output=5):
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

    clf.add(Dense(units=512, activation='relu'))
    clf.add(Dense(units=1024, activation='relu'))

    clf.add(Dense(units=n_output, activation='softmax'))
    clf.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])  # TODO: add AUC metric; add class weights

    return clf


def run_training(train, model_dir, dataset):
    if dataset.NAME == "MITBIH":
        train = balance_data(train)
    print(dataset.NAME)
    X_train, y_train = prepare_data(train)
    clf = create_model(X_train.shape[1])
    clf.fit(X_train, y_train, validation_split=0.1, epochs=10, shuffle=True)
    clf.save(model_dir)


def run_testing(test, model_dir):
    X_test, y_test = prepare_data(test)
    clf = load_model(model_dir)
    loss, acc = clf.evaluate(X_test, y_test)
    y_pred = clf.predict_classes(X_test)
    return y_pred, loss, acc


def run(dataset):
    train, test = dataset.load_data(Config.data_dir)
    run_training(train, Config.model_dir, dataset)
    run_testing(test, Config.model_dir)
