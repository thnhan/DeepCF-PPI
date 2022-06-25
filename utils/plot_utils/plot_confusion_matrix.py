import numpy as np
import matplotlib.pyplot as plt
import itertools
from my_tools import get_scores


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label_new')
    plt.xlabel('Predicted label_new')


def plot_average_confusion_matrix(arr_cm, classes=[-1, 1],
                                  normalize=True,
                                  title='Confusion matrix',
                                  cmap='Blues_r'):
    # Calculate average value of arr_cm. arr_cm is array of cm (Confusion matrix)
    n_folds = len(arr_cm)
    ave_cm = np.zeros_like(arr_cm[0])
    for cm in arr_cm:                              
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        ave_cm = ave_cm + cm
    ave_cm /= n_folds

    plot_confusion_matrix(ave_cm, classes=classes, normalize=True,
                      title='Normalized confusion matrix')


def plot_average_confusion_matrix_(arr_y_test, arr_y_pred, classes=[0, 1], normalize=True, title='Normalized confusion matrix'):
    arr_cm = get_scores.calc_confusion_matrix_KFOLD(arr_y_test, arr_y_pred)
    plot_average_confusion_matrix(arr_cm,
                                        classes=classes,
                                        normalize=normalize,            
                                        title=title)


# import matplotlib.pyplot as plt
# import itertools
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label_new')
#     plt.xlabel('Predicted label_new')
#
# # Plot non-normalized confusion matrix
# class_names = [0, 1, 2]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()