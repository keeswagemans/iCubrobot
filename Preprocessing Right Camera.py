# PIPELINE RIGHT CAMERA
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

# Path
path = '/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/'
path_right_camera = path + 'icub_right/'

# Needed variables
img_ids = np.arange(72)
objects = open(path + 'objects.txt')
objects = [i.split('\n', 1)[0] for i in objects]
objects = [i.replace(' ', '_') for i in objects]

# Function for loading the data

def load_data():

    # Creating lists
    X_train_right = []
    X_val_right = []
    X_test_right = []

    Y_train_right = []
    Y_val_right = []
    Y_test_right = []

    # Needed variables
    nof_img = 72
    nof_objects = 210

    # Looping through all the objects
    for i in range(nof_objects):
        obj_path = path_right_camera + objects[i] + '/' + objects[i]
        randomized_ids = np.random.choice(np.arange(nof_img), nof_img, replace=False)
        training_ids, validation_ids, testing_ids = randomized_ids[0:42], randomized_ids[42:57], randomized_ids[57:72]
        print(testing_ids)
        print(obj_path)

        # Looping through all the images
        for ii in range(nof_img):
            image = cv2.imread(obj_path + '_color_' + str(ii) + '.png')
            image = image[70:300, 160:400]
            image = cv2.resize(image, (64,64))

            if ii in training_ids:
                X_train_right.append(image)
                Y_train_right.append(i)

            if ii in validation_ids:
                X_val_right.append(image)
                Y_val_right.append(i)

            if ii in testing_ids:
                X_test_right.append(image)
                Y_test_right.append(i)

    # Turning into numpy array
    X_train_right = np.asarray(X_train_right)
    X_val_right = np.asarray(X_val_right)
    X_test_right = np.asarray(X_test_right)
    Y_train_right = np.asarray(Y_train_right)
    Y_val_right = np.asarray(Y_val_right)
    Y_test_right = np.asarray(Y_test_right)

    return X_train_right, X_val_right, X_test_right, Y_train_right, Y_val_right, Y_test_right

X_train_right, X_val_right, X_test_right, Y_train_right, Y_val_right, Y_test_right = load_data()

Y_train_right = to_categorical(Y_train_right, dtype='float32')
Y_val_right = to_categorical(Y_val_right, dtype='float32')
Y_test_right = to_categorical(Y_test_right, dtype='float32')

# Display image
def display_rand_img(images):
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.show()

display_rand_img(X_val_right)

# Pickle
pickle.dump(X_train_right, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_train_right.pkl', 'wb'))
pickle.dump(X_val_right, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_val_right.pkl', 'wb'))
pickle.dump(X_test_right, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_right.pkl', 'wb'))

pickle.dump(Y_train_right, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_train_right.pkl', 'wb'))
pickle.dump(Y_val_right, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_val_right.pkl', 'wb'))
pickle.dump(Y_test_right, open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_right.pkl', 'wb'))

print(Y_test_right.shape)