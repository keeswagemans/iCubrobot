# Loading libraries
import pickle
import numpy as np

# Load pickle

def load_train_images():
    X_train_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_train_left.pkl', 'rb'))
    Y_train_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_train_left.pkl', 'rb'))
    X_train_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_train_realsense.pkl', 'rb'))
    Y_train_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_train_realsense.pkl', 'rb'))
    X_train_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_train_right.pkl', 'rb'))
    Y_train_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_train_right.pkl', 'rb'))
    return X_train_left, Y_train_left, X_train_realsense, Y_train_realsense, X_train_right, Y_train_right

X_train_left, Y_train_left, X_train_realsense, Y_train_realsense, X_train_right, Y_train_right = load_train_images()

def load_test_images():
    X_test_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_left.pkl', 'rb'))
    Y_test_left = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_left.pkl', 'rb'))
    X_test_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_realsense.pkl', 'rb'))
    Y_test_realsense = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_realsense.pkl', 'rb'))
    X_test_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_right.pkl', 'rb'))
    Y_test_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_right.pkl', 'rb'))
    return X_test_left, Y_test_left, X_test_realsense, Y_test_realsense, X_test_right, Y_test_right

X_test_left, Y_test_left, X_test_realsense, Y_test_realsense, X_test_right, Y_test_right = load_test_images()

# Ensemble model

import tensorflow as tf

model_left = tf.keras.models.load_model('CNN_1_left')
model_realsense = tf.keras.models.load_model('CNN_2_realsense')
model_right = tf.keras.models.load_model('CNN_3_right')

pred1 = model_left.predict(X_test_left)
pred2 = model_realsense.predict(X_test_realsense)
pred3 = model_right.predict(X_test_right)

finalpred = (pred1+pred2+pred3)/3

y_pred = np.argmax(finalpred, axis=-1)
y_true = np.argmax(Y_test_right, axis=-1)
print(finalpred)

print(Y_test_right.shape)

# Confusion Matrix

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sbn

CM = confusion_matrix(y_true, y_pred)
ax = plt.axes()
sbn.heatmap(CM, annot = True,
            annot_kws={"size":10},
            xticklabels = np.arange(0,210),
            yticklabels = np.arange(0,210), ax = ax)
ax.set_title("Confusion Matrix for Test Data")
plt.show()

# Classification Report

from sklearn.metrics import classification_report

report_ensemble_model = classification_report(y_true, y_pred)
print(report_ensemble_model)

# ROC-curve

path = '/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/'
objects = open(path + 'objects.txt')
objects = list(objects)
objects = [i.split('\n', 1)[0] for i in objects]
objects = [i.replace(' ', '_') for i in objects]

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

fig, c_ax = plt.subplots(1, 1, figsize = (12,8))

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(objects):
        fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), y_pred[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)


print('ROC AUC score:', multiclass_roc_auc_score(y_true, y_pred))

c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.show()

