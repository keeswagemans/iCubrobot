# REALSENSE CAMERA
# DATA PREPROCESSING

# Importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import pickle
import random
import time

np.random.seed(200)
random_nr = np.random.randint(10)
if random_nr == 9:
    print('Randomization Correct')
else:
    print('Randomization Failed')

# Making a path
path = '/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/'
path_realsense_camera = path + 'realsense/'

# Extracting labels
objects = open(path + 'objects.txt')
objects = list(objects)
objects = [i.split('\n', 1)[0] for i in objects]
objects = [i.replace(' ', '_') for i in objects]

# Creating function for loading the data

def load_data():

    # Creating lists
    X_train_realsense = []
    X_val_realsense = []
    X_test_realsense = []

    Y_train_realsense = []
    Y_val_realsense = []
    Y_test_realsense = []

    # Needed variables
    nof_img = 72
    nof_objects = 210

    # Looping through all the objects
    for i in range(nof_objects):
        obj_path = path_realsense_camera + objects[i] + '/' + objects[i]
        randomized_ids = np.random.choice(np.arange(nof_img), nof_img, replace=False)
        training_ids, validation_ids, testing_ids = randomized_ids[0:42], randomized_ids[42:57], randomized_ids[57:72]
        print(obj_path)
        print(testing_ids)

        # Looping through all the images
        for ii in range(nof_img):
            image = cv2.imread(obj_path + '_color_' + str(ii) + '.png')
            image = image[70:380, 180:440]
            image = cv2.resize(image, (64,64))

            if ii in training_ids:
                X_train_realsense.append(image)
                Y_train_realsense.append(i)

            if ii in validation_ids:
                X_val_realsense.append(image)
                Y_val_realsense.append(i)

            if ii in testing_ids:
                X_test_realsense.append(image)
                Y_test_realsense.append(i)

    # Turning into numpy array
    X_train_realsense = np.asarray(X_train_realsense)
    X_val_realsense = np.asarray(X_val_realsense)
    X_test_realsense = np.asarray(X_test_realsense)
    Y_train_realsense = np.asarray(Y_train_realsense)
    Y_val_realsense = np.asarray(Y_val_realsense)
    Y_test_realsense = np.asarray(Y_test_realsense)

    return X_train_realsense, X_val_realsense, X_test_realsense, Y_train_realsense, Y_val_realsense, Y_test_realsense

X_train_realsense, X_val_realsense, X_test_realsense, Y_train_realsense, Y_val_realsense, Y_test_realsense = load_data()

Y_train_realsense = to_categorical(Y_train_realsense, dtype='float32')
Y_val_realsense = to_categorical(Y_val_realsense, dtype='float32')
Y_test_realsense = to_categorical(Y_test_realsense, dtype='float32')

# Function for displaying images
def display_rand_img(images):
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.show()

display_rand_img(X_val_realsense)

# Pickle
pickle.dump(X_train_realsense, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_train_realsense.pkl', 'wb'))
pickle.dump(X_val_realsense, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_val_realsense.pkl', 'wb'))
pickle.dump(X_test_realsense, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_realsense.pkl', 'wb'))

pickle.dump(Y_train_realsense, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_train_realsense.pkl', 'wb'))
pickle.dump(Y_val_realsense, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_val_realsense.pkl', 'wb'))
pickle.dump(Y_test_realsense, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_realsense.pkl', 'wb'))

print(Y_train_realsense.shape)


