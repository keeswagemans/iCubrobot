# Importing libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Loading pickle
def load_images():
    X_train_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_train_realsense.pkl', 'rb'))
    X_val_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_val_realsense.pkl', 'rb'))
    X_test_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_realsense.pkl', 'rb'))
    return X_train_realsense, X_val_realsense, X_test_realsense

X_train_realsense, X_val_realsense, X_test_realsense = load_images()

def load_labels():
    Y_train_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_train_realsense.pkl', 'rb'))
    Y_val_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_val_realsense.pkl', 'rb'))
    Y_test_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_realsense.pkl', 'rb'))
    return Y_train_realsense, Y_val_realsense, Y_test_realsense

Y_train_realsense, Y_val_realsense, Y_test_realsense = load_labels()

print(Y_test_realsense.shape)
print(X_test_realsense.shape)

# Training
import tensorflow as tf

input_shape_realsense = X_train_realsense[0].shape

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=input_shape_realsense),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation = 'relu', padding = 'same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(210, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(X_train_realsense, Y_train_realsense, epochs=20, batch_size=64, validation_data=(X_val_realsense, Y_val_realsense))

# Loss and accuracy

test_loss, test_acc = model.evaluate(X_test_realsense, Y_test_realsense, verbose=2)
print("Test accuracy: " + str(test_acc))

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

# Saving the model for ensemble learning

model.save('CNN_2_realsense', save_format='h5')