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

print(X_train_left.shape)
print(Y_train_left.shape)

input_shape_cam_left = X_train_left[0].shape

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape = (64, 64, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2))
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2))
    tf.keras.layers.Dense(210, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(X_train_left, Y_train_left, epochs=20, batch_size=64, validation_data=(X_val_left, Y_val_left))

# Loss and accuracy
def summarize_diagnostics(history):
    # loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='yellow', label='validation')

    # accuracy
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='yellow', label='validation')
    plt.legend(['training', 'validation'])
    plt.show()

summarize_diagnostics(history)

# Saving the model

model.save('CNN_1_left', save_format='h5')


