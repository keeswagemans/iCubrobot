# Importing libraries
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sbn

# Loading data
def load_images():
    X_train_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_train_right.pkl', 'rb'))
    X_val_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_val_right.pkl', 'rb'))
    X_test_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/X_test_right.pkl', 'rb'))
    return X_train_right, X_val_right, X_test_right

X_train_right, X_val_right, X_test_right = load_images()

def load_labels():
    Y_train_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_train_right.pkl', 'rb'))
    Y_val_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_val_right.pkl', 'rb'))
    Y_test_right = pickle.load(open('/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/Y_test_right.pkl', 'rb'))
    return Y_train_right, Y_val_right, Y_test_right

Y_train_right, Y_val_right, Y_test_right = load_labels()

# Loading model

model = tf.keras.models.load_model('CNN_3_right')

# Evaluation
test_loss, test_acc = model.evaluate(X_test_right, Y_test_right, verbose=2)
print("Test accuracy: " + str(test_acc))

# Make label from one-hot encoding for confusion matrices
Y_train_right = np.argmax(Y_train_right, axis=1)
Y_val_right = np.argmax(Y_val_right, axis=1)
Y_test_right = np. argmax(Y_test_right, axis=1)

# print(Y_train_right.shape)
#
# # Confusion Matrix for Validation Data
# path = '/Volumes/Macintosh HD/Users/keeswagemans/Desktop/Thesis/iCub dataset/'
# objects = open(path + 'objects.txt')
# objects = list(objects)
# objects = [i.split('\n', 1)[0] for i in objects]
# objects = [i.replace(' ', '_') for i in objects]
#
# predictions_val_right = model.predict(X_val_right)
# pred_val_labels_right = np.argmax(predictions_val_right, axis=1)
#
# CM1_1 = confusion_matrix(Y_val_right, pred_val_labels_right, labels=np.arange(0,70))
# ax = plt.axes()
# sbn.heatmap(CM1_1, annot = False,
#             annot_kws = {"size":10},
#             xticklabels = np.arange(0,70,10),
#             yticklabels = np.arange(0,70,10), ax=ax),
# ax.set_title("CM for Classes 0 - 70 of Validation Data")
# plt.xticks(ticks=np.arange(0,70,10), labels=np.arange(0,70,10))
# plt.yticks(ticks=np.arange(0,70,10), labels=np.arange(0,70,10))
# plt.show()
#
# CM1_2 = confusion_matrix(Y_val_right, pred_val_labels_right, labels=np.arange(70,140))
# ax = plt.axes()
# sbn.heatmap(CM1_2, annot = False,
#             annot_kws = {"size":10},
#             xticklabels = np.arange(70,140,10),
#             yticklabels = np.arange(70,140,10), ax=ax),
# ax.set_title("CM for Classes 70 - 140 of Validation Data")
# plt.xticks(ticks=np.arange(0,70,10), labels=np.arange(70,140,10))
# plt.yticks(ticks=np.arange(0,70,10), labels=np.arange(70,140,10))
# plt.show()
#
# CM1_3 = confusion_matrix(Y_val_right, pred_val_labels_right, labels = np.arange(140, 210))
# ax = plt.axes()
# sbn.heatmap(CM1_3, annot = False,
#             annot_kws = {"size":10},
#             xticklabels = np.arange(140,210,10),
#             yticklabels = np.arange(140,210,10),
#             ax = ax),
# ax.set_title("CM for Classes 140 - 210 of Validation Data")
# plt.xticks(np.arange(0,70,10), labels=np.arange(140,210,10))
# plt.yticks(np.arange(0,70,10), labels=np.arange(140,210,10))
# plt.show()

# Confusion Matrix for Test Data

predictions_test_right = model.predict(X_test_right)
pred_test_labels_right = np.argmax(predictions_test_right, axis=-1)

print(pred_test_labels_right[1950:1965])
#
# CM2_1 = confusion_matrix(Y_test_right, pred_test_labels_right, labels = np.arange(0,70))
# ax = plt.axes()
# sbn.heatmap(CM2_1, annot=False,
#             annot_kws={"size":10},
#             xticklabels=np.arange(0,70,10),
#             yticklabels=np.arange(0,70,10), ax=ax)
# ax.set_title("CM for Classes 0 - 70 of Test Data")
# plt.xticks(np.arange(0,70,10), labels=np.arange(0,70,10))
# plt.yticks(np.arange(0,70,10), labels=np.arange(0,70,10))
# plt.show()
#
# CM2_2 = confusion_matrix(Y_test_right, pred_test_labels_right, labels = np.arange(70,140))
# ax = plt.axes()
# sbn.heatmap(CM2_2, annot=False,
#             annot_kws={"size":10},
#             xticklabels = np.arange(70,140,10),
#             yticklabels = np.arange(70,140,10), ax=ax)
# ax.set_title("CM for Classes 70 - 140 of Test Data")
# plt.xticks(np.arange(0,70,10), labels=np.arange(70,140,10))
# plt.yticks(np.arange(0,70,10), labels=np.arange(70,140,10))
# plt.show()
#
# CM2_3 = confusion_matrix(Y_test_right, pred_test_labels_right, labels = np.arange(140,210))
# ax = plt.axes()
# sbn.heatmap(CM2_3, annot=False,
#             annot_kws={"size":10},
#             xticklabels = np.arange(140,210,10),
#             yticklabels = np.arange(140,210,10), ax = ax)
# ax.set_title("CM for Classes 140 - 210 of Test Data")
# plt.xticks(np.arange(0,70,10), labels=np.arange(140,210,10))
# plt.yticks(np.arange(0,70,10), labels=np.arange(140,210,10))
# plt.show()

# Classification Report for Validation Data

# report_val = classification_report(Y_val_right, pred_val_labels_right)
# print(report_val)
#
# # Classification Report for Test Data
#
# report_test = classification_report(Y_test_right, pred_test_labels_right, zero_division=1)
# print(report_test)

# # ROC-curve
#
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.metrics import roc_curve, auc, roc_auc_score
#
#
# # fig, c_ax = plt.subplots(1, 1, figsize = (12,8))
# #
# # def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
# #     lb = LabelBinarizer()
# #     lb.fit(y_test)
# #     y_test = lb.transform(y_test)
# #     y_pred = lb.transform(y_pred)
# #
# #     for (idx, c_label) in enumerate(objects):
# #         fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), y_pred[:, idx])
# #         c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
# #     c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')
# #     return roc_auc_score(y_test, y_pred, average=average)
# #
# #
# # print('ROC AUC score:', multiclass_roc_auc_score(Y_test_right, pred_test_labels_right))
# #
# # c_ax.legend()
# # c_ax.set_xlabel('False Positive Rate')
# # c_ax.set_ylabel('True Positive Rate')
# # plt.show()
#
# # SECOND ROC
#
# from tensorflow.keras.utils import to_categorical
#
# Y_test_right = to_categorical(Y_test_right, dtype='float32')
# pred_test_labels_right = to_categorical(pred_test_labels_right, dtype ='float32')
#
# from sklearn.metrics import roc_curve, auc
#
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(210):
#     fpr[i], tpr[i], _ = roc_curve(Y_test_right[:, i], pred_test_labels_right[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_right.ravel(), pred_test_labels_right.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # Plot of a ROC curve for a specific class
# plt.figure()
# plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
#
# # Plot ROC curve
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]))
# for i in range(210):
#     plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
#                                    ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('One-versus-Rest ROC')
# plt.legend(loc="lower right", ncols=5, fontsize='xx-small', bbox_to_anchor=(3, 0))
# plt.show()
#
#


