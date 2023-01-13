# LEFT CAMERA
# DATA PREPROCESSING

# Importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
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
path_left_camera = path + 'icub_left/'

# Making function for appending to X_train, X_val, X_test, Y_train, Y_val, Y_test
objects = open(path + 'objects.txt')
objects = list(objects)
objects = [i.split('\n', 1)[0] for i in objects]
objects = [i.replace(' ', '_') for i in objects]

# Creating function for loading the data

def load_data():

    # Creating lists
    X_train_left = []
    X_val_left = []
    X_test_left = []

    Y_train_left = []
    Y_val_left = []
    Y_test_left = []

    # Needed variables
    nof_img = 72
    nof_objects = 210

    # Looping through all the objects
    for i in range(nof_objects):
        obj_path = path_left_camera + objects[i] + '/' + objects[i]
        randomized_ids = np.random.choice(np.arange(nof_img), nof_img, replace=False)
        training_ids, validation_ids, testing_ids = randomized_ids[0:42], randomized_ids[42:57], randomized_ids[57:72]
        print(testing_ids)
        print(obj_path)

        # Looping through all the images
        for ii in range(nof_img):
            image = cv2.imread(obj_path + '_color_' + str(ii) + '.png')
            image = image[70:300, 160:400]
            image = cv2.resize(image, (64, 64))

            if ii in training_ids:
                X_train_left.append(image)
                Y_train_left.append(i)

            if ii in validation_ids:
                X_val_left.append(image)
                Y_val_left.append(i)

            if ii in testing_ids:
                X_test_left.append(image)
                Y_test_left.append(i)

    # Turning into numpy array
    X_train_left = np.asarray(X_train_left)
    X_val_left = np.asarray(X_val_left)
    X_test_left = np.asarray(X_test_left)
    Y_train_left = np.asarray(Y_train_left)
    Y_val_left = np.asarray(Y_val_left)
    Y_test_left = np.asarray(Y_test_left)

    return X_train_left, X_val_left, X_test_left, Y_train_left, Y_val_left, Y_test_left

X_train_left, X_val_left, X_test_left, Y_train_left, Y_val_left, Y_test_left = load_data()

Y_train_left = to_categorical(Y_train_left, dtype='float32')
Y_val_left = to_categorical(Y_val_left, dtype='float32')
Y_test_left = to_categorical(Y_test_left, dtype='float32')

input_shape_cam_left = X_train_left[0].shape


def reshape_input_vect(vect, size=input_shape_cam_left):
    inp_vect = []
    for i in range(vect.shape[0]):
        inp_vect.append(np.reshape(vect[i], input_shape_cam_left))
    return np.asarray(inp_vect)

X_train_left, X_val_left, X_test_left = reshape_input_vect(X_train_left), reshape_input_vect(X_val_left), reshape_input_vect(X_test_left)


# Function for displaying images

def display_rand_img(images):
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.show()

display_rand_img(X_val_left)

# Save as pickle
pickle.dump(X_train_left, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_train_left.pkl', 'wb'))
pickle.dump(X_val_left, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_val_left.pkl', 'wb'))
pickle.dump(X_test_left, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_left.pkl', 'wb'))

pickle.dump(Y_train_left, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_train_left.pkl', 'wb'))
pickle.dump(Y_val_left, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_val_left.pkl', 'wb'))
pickle.dump(Y_test_left, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_left.pkl', 'wb'))

print(X_train_left.shape)