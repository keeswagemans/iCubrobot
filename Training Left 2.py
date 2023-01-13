# TRAINING MODEL LEFT CAMERA
# Importing libraries

import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# Loading pickle

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

# Training

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(352, activation='relu'),
    tf.keras.layers.Dense(384, activation='sigmoid'),
    tf.keras.layers.Dense(224, activation='relu'),
    tf.keras.layers.Dense(210, activation= 'softmax')])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(X_train_left, Y_train_left, epochs=20, batch_size=64, validation_data=(X_val_left, Y_val_left))

model.save('CNN_1_left_2', format='h5')
