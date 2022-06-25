"""
@author: thnhan
"""
import itertools

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc, confusion_matrix, roc_auc_score
# from utils.reportscore.my_metrics import calculate_scores
from sklearn.metrics import (plot_confusion_matrix,
                             plot_precision_recall_curve,
                             plot_roc_curve)
from matplotlib import cm
from matplotlib import pyplot as plt


def calculate_scores(true_y, prob_y, pred_y):
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)
    prob_y = np.array(prob_y)

    true_y[true_y == -1] = true_y[true_y == -1] + 1
    pred_y[pred_y == -1] = pred_y[pred_y == -1] + 1
    # print(true_y)
    # print(pred_y)
    test_num = len(true_y)
    e = 1.0e-6

    cm = confusion_matrix(true_y, pred_y)  # sklearn lib
    # cm.astype(float)
    tn, fp, fn, tp = cm.ravel()

    scores = dict()
    scores.update({'tn': tn,
                   'fp': fp,
                   'fn': fn,
                   'tp': tp})
    tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)
    tpr = tp / (tp + fn + e)
    tnr = tn / (tn + fp + e)
    fpr = 1 - tnr  # fp / (fp + tn + e)
    fnr = 1 - tpr

    scores.update({'Accuracy': (tp + tn) / (test_num + e)})

    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + e)
    scores.update({'MCC-score': mcc})

    precision = tp / (tp + fp + e)
    scores.update({'Precision': precision})

    f1 = (tp * 2) / (tp * 2 + fp + fn + e)
    scores.update({'F1-score': f1})

    sensitivity = tp / (tp + fn + e)
    scores.update({'Sensitivity': sensitivity})
    scores.update({'Recall': sensitivity})  # recall = sensitivity

    specificity = tn / (tn + fp + e)
    scores.update({'Specificity': specificity})

    npv = tn / (tn + fn + e)
    scores.update({'NPV': npv})
    scores.update({'PPV': precision})
    # scores.update({'AUC': roc_auc_score(test_y, prob_y[:, 1])})
    fpr, tpr, _ = roc_curve(true_y, prob_y[:, 1])
    scores.update({'AUC': roc_auc_score(true_y, prob_y[:, 1])})
    scores.update({'AUPR': average_precision_score(true_y, prob_y[:, 1])})

    # scores['fpr'] = fpr
    # scores['fnr'] = fnr
    # scores['f1'] = f1
    # scores['tpr'] = = scores['sensitivity'] = tpr
    # scores['tnr'] = scores['specificity'] = tnr

    return scores


def print_metrics(true_y, prob_y=None, pred_y=None, metrics=None, verbose=1):
    if pred_y is None:
        pred_y = np.argmax(prob_y, axis=1)
    scores = calculate_scores(true_y, prob_y, pred_y)

    if metrics is None:
        metrics = ['Accuracy',
                   'Sensitivity',
                   'Specificity',
                   'Precision',
                   'NPV',
                   'F1-score',
                   'MCC-score',
                   'AUC',
                   'AUPR']

    fmt = metrics + list(map(lambda k: scores[k], metrics))
    txt = '|{:12}' * len(metrics) + '|' + '\n'
    txt = txt + '|------------' * len(metrics) + '|' + '\n'
    txt = txt + '|{:12.2%}' + '|{:12.2%}' * (len(metrics) - 1) + '|'
    if verbose == 1:
        print(txt.format(*fmt))

    return scores


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


def plot_metrics(model, test_X, test_y):
    plot_confusion_matrix(model, test_X, test_y,
                          normalize='true',
                          cmap=cm.cmap_d['Blues'])
    # plot_confusion_matrix
    plot_precision_recall_curve(model, test_X, test_y)
    plot_roc_curve(model, test_X, test_y)

    return None


def my_cv_report(cv_scores, metrics: list = None, verbose=1):
    from numpy import mean, std

    if metrics is None:
        metrics = ['Accuracy',
                   'Sensitivity',
                   'Specificity',
                   'Precision',
                   'NPV',
                   'F1-score',
                   'MCC-score',
                   'AUC',
                   'AUPR']

    tam = [list(map(lambda k: scores[k], metrics)) for scores in cv_scores]
    avg = mean(tam, axis=0).tolist()
    std = std(tam, axis=0).tolist()
    txt = '|{:15}' * len(metrics) + '|' + '\n'
    txt = txt + '|---------------' * len(metrics) + '|' + '\n'
    txt = txt + '|{:8.2%}+/-{:0.2%}' + '|{:8.2%}+/-{:0.2%}' * (len(metrics) - 1) + '|'  # + '\n'
    # txt = txt + '|+/-{:12.2%}' + '|{:12.2%}' * (len(metrics) - 1) + '|' + '\n'
    temp = []
    for a, b in zip(avg, std):
        temp.append(a)
        temp.append(b)
    fmt = metrics + temp
    if verbose == 1:
        print(txt.format(*fmt))

    cols = metrics[:]
    rows = ['fold_' + str(i + 1) for i in range(len(cv_scores))]
    data = []
    for fold in cv_scores:
        data.append([fold[c] for c in cols])

    return my_summary_table(data, cols, rows)


def my_summary_table(data, cols, rows):
    import pandas as pd
    return pd.DataFrame(data=data, columns=cols, index=rows)


if __name__ == "__main__":
    print("-" * 15)
