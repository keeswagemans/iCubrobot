import numpy as np
import pickle
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam

import keras_tuner
from keras_tuner.tuners import RandomSearch

from tensorflow.keras.utils import to_categorical


# Loading pickles
def load_images():
    X_train_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_train_left.pkl', 'rb'))
    X_val_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_val_left.pkl', 'rb'))
    X_test_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_left.pkl', 'rb'))
    return X_train_left, X_val_left, X_test_left


X_train_left, X_val_left, X_test_left = load_images()


def load_labels():
    Y_train_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_train_left.pkl', 'rb'))
    Y_val_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_val_left.pkl', 'rb'))
    Y_test_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_left.pkl', 'rb'))
    return Y_train_left, Y_val_left, Y_test_left


Y_train_left, Y_val_left, Y_test_left = load_labels()

input_shape_cam_left = X_train_left[0].shape


# Random search
def build_model(hp):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    for i in range(hp.Int('layers', 2, 6)):  # Searching through for 2, 3 and 4 hidden layers
        model.add(tf.keras.layers.Dense(
            units=hp.Int('units_' + str(i), 64, 512, step=32),
            # Searching for optimal amount of nodes per hidden layer from 32 to 512 with step size of 32
            activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid'])))  # Searching for optimal activation method
    model.add(tf.keras.layers.Dense(4,
                                    activation='softmax'))  # Output layer is kept out of the for loop because that is fixed
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[0.1, 1e-2, 1e-3, 1e-4])),  # Learning Rate of 0.01, 0.001 and 0.0001
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',  # Objective is to maximize validation accuracy
    max_trials=10,  # Trial 5 times
    executions_per_trial=3,  # Each trial, try 1 different model
)

tuner.search_space_summary()

tuner.search(X_train_left, Y_train_left,
             epochs=10,
             validation_data=(X_val_left, Y_val_left))

tuner.results_summary()

