import itertools

import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as skmetrics


def report_performance(scores_array):
    print(("accuracy=%.2f%% (+/- %.2f%%)" % (
        np.mean(scores_array, axis=0)[0] * 100, np.std(scores_array, axis=0)[0] * 100)))
    print(("precision=%.2f%% (+/- %.2f%%)" % (
        np.mean(scores_array, axis=0)[1] * 100, np.std(scores_array, axis=0)[1] * 100)))
    print(
        "recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2] * 100, np.std(scores_array, axis=0)[2] * 100))
    print("specificity=%.2f%% (+/- %.2f%%)" % (
        np.mean(scores_array, axis=0)[3] * 100, np.std(scores_array, axis=0)[3] * 100))
    print("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4] * 100, np.std(scores_array, axis=0)[4] * 100))
    print("f1_score=%.2f%% (+/- %.2f%%)" % (
        np.mean(scores_array, axis=0)[5] * 100, np.std(scores_array, axis=0)[5] * 100))
    print(
        "roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6] * 100, np.std(scores_array, axis=0)[6] * 100))
    print(
        "roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7] * 100, np.std(scores_array, axis=0)[7] * 100))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, show=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if show:
        plt.show()


def report_fold(true_label, prob_label, fmt="%0.4f"):
    true_label = np.argmax(true_label, axis=1)
    pred_label = np.argmax(prob_label, axis=1)
    cm = skmetrics.confusion_matrix(true_label, pred_label)
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)  # TP / (TP + FP)
    precision = cm[1, 1] / np.sum(cm[:, 1])  # TP / (TP + FN)
    recall = cm[1, 1] / np.sum(cm[1, :])
    specificity = cm[0, 0] / np.sum(cm[:, 0])# TN / (TN+FP)
    f1_score = skmetrics.f1_score(true_label, pred_label)
    mcc_score = skmetrics.matthews_corrcoef(true_label, pred_label)
    auc_score = 0
    pr_score = 0
    if len(np.unique(true_label)) == 2:
        auc_score = skmetrics.roc_auc_score(true_label, prob_label[:, 1])
        pr_score = skmetrics.average_precision_score(true_label, prob_label[:, 1])

    print("\tacc {.4f}\t", fmt % accuracy)
    print("pre:   ", fmt % precision)
    print("rec:      ", fmt % recall)
    print("spec: ", fmt % specificity)
    print("MCC:         ", fmt % mcc_score)
    print("AUC:         ", fmt % auc_score)
    print("AUPR:        ", fmt % pr_score)
    print("F1 Score:    ", fmt % f1_score)

    plot_confusion_matrix(cm, classes=['0', '1'], normalize=True, show=False)

    return cm, accuracy, precision, recall, specificity, mcc_score, auc_score, f1_score


if __name__ == "__main__":
    prob = np.array([.8, .7, .4, .3, .1, .1])
    prob = np.vstack((np.ones(6, ) - prob, prob)).transpose()
    print(prob)
    report_fold([1, 1, 1, 0, 0, 0], prob)
