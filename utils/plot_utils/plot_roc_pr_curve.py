"""
@author: thnhan
ref: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
"""

import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc
)


def plot_folds(plt, arr_true_y: list, arr_prob_y: list):
    # init
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    ax_roc = ax[0]
    ax_rpc = ax[1]

    n_samples = 0
    for i, y_test in enumerate(arr_true_y):
        n_samples += y_test.shape[0]
    mean_fpr = np.linspace(0, 1, n_samples)
    roc_aucs = []
    tprs = []

    # get fpr, tpr scores
    for i, (y_test, y_prob) in enumerate(zip(arr_true_y, arr_prob_y)):
        # # print(y_test)
        # if np.ndim(y_prob) > 1:
        #     y_prob = y_prob[:, 1]  # only use prob of class 1
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        # plot ROC curve
        ax_roc.plot(fpr, tpr, lw=1, alpha=0.5, label='Fold %d (AUC = %0.3f)' % (i + 1, roc_auc))

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_aucs.append(roc_auc)

    ax_roc.plot([0, 1], [0, 1], lw=2, alpha=.8, linestyle='--', color='r', label='Chance')

    # Ve ROC mean
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    std_roc_auc = np.std(roc_aucs)
    ax_roc.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean (AUC = %0.3f $\pm$ %0.2f)' % (mean_roc_auc, std_roc_auc),
                lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                        alpha=.2)  # ,label_new=r'$\pm$ 1 std. dev.')

    # Dat ten 
    ax_roc.set(
        xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic"
    )
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")

    ####################################################################################
    # mean_recall = np.linspace(0, 1, n_samples)
    # pres = []; rpc_aucs = []
    # get precision, recall scores
    for i, (y_test, y_prob) in enumerate(zip(arr_true_y, arr_prob_y)):
        # if np.ndim(y_prob) > 1:
        #     y_prob = y_prob[:, 1]  # only use prob of class 1
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        average_precision = average_precision_score(y_test,
                                                    y_prob)  # Quay lại sử dụng lệnh này nếu có sai sót
        # plot precision recall curve

        ax_rpc.plot(
            recall, precision,
            lw=1, alpha=0.5, label='Fold {:2d} (AP = {:.3f})'.format(
                i + 1, average_precision
            )
        )

        # interp_precision = np.interp(mean_recall, recall, precision)
        # # interp_precision[0] = 0.0
        # pres.append(interp_precision)
        # rpc_aucs.append(rpc_auc)

    y_tests = np.array([])
    for y_test in arr_true_y:
        y_tests = np.hstack((y_tests, y_test.ravel()))

    no_skill = len(y_tests[y_tests == 1]) / y_tests.shape[0]
    ax_rpc.plot(
        [0, 1], [no_skill, no_skill],
        linestyle='--', lw=2, color='r', label='Chance'
    )

    # Ve duong mean
    all_y_test = np.concatenate(arr_true_y)
    all_y_prob = np.concatenate(arr_prob_y)
    precision, recall, _ = precision_recall_curve(all_y_test, all_y_prob)

    ax_rpc.plot(
        recall, precision, color='b',
        label=r'Overall (AP = %0.3f)' %
              (average_precision_score(all_y_test, all_y_prob)),
        lw=2, alpha=.8
    )

    # Dat ten
    ax_rpc.set_title('Recall Precision Curve')
    ax_rpc.set_xlabel('Recall')
    ax_rpc.set_ylabel('Precision')
    ax_rpc.legend(loc="lower left")


def plot_methods(methods_name_and_y_prob: dict, save=None):
    """
    @author: thnhan

    Parameters:
    ==========================

    `methods_name_and_y_prob`: `dict('method name', [y_true, y_prob])`
    """
    import matplotlib.pyplot as plt

    # init
    name_methods = list(methods_name_and_y_prob.keys())
    tam = list(methods_name_and_y_prob.values())
    # print(methods_name_and_y_prob)
    arr_y_test = [tam[i][0] for i in range(len(methods_name_and_y_prob))]
    arr_y_prob = [tam[i][1] for i in range(len(methods_name_and_y_prob))]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    ax_roc = ax[0]
    ax_rpc = ax[1]

    # get fpr, tpr scores
    for i, (name_method, y_test, y_prob) in enumerate(zip(name_methods, arr_y_test, arr_y_prob)):
        # if np.ndim(y_prob) > 1:
        #     y_prob = y_prob[:, 1]  # only use prob of class 1
        fpr, tpr, _ = roc_curve(y_test.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)
        # plot ROC curve
        ax_roc.plot(
            fpr, tpr,
            lw=1.5, alpha=1,
            label=name_method + ' (AUC = %0.3f)' % roc_auc
        )

        # interp_tpr = np.interp(mean_fpr, fpr, tpr)
        # interp_tpr[0] = 0.0
        # tprs.append(interp_tpr)
        # roc_aucs.append(roc_auc)

    ax_roc.plot([0, 1], [0, 1], lw=2, alpha=.8, linestyle='--', color='r')  # , label_new='Chance')

    # Dat ten 
    ax_roc.set(
        xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic"
    )
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")

    ####################################################################################
    # mean_recall = np.linspace(0, 1, n_samples)
    # pres = []; rpc_aucs = []
    # get precision, recall scores
    for i, (name_method, y_test, y_prob) in enumerate(zip(name_methods, arr_y_test, arr_y_prob)):
        # if np.ndim(y_prob) > 1:
        #     y_prob = y_prob[:, 1]  # only use prob of class 1
        precision, recall, _ = precision_recall_curve(y_test.ravel(), y_prob.ravel())
        average_precision = average_precision_score(y_test,
                                                    y_prob)  # Quay lại sử dụng lệnh này nếu có sai sót
        # plot precision recall curve

        ax_rpc.plot(
            recall, precision,
            lw=1.5, alpha=1,
            label=name_method + ' (AUPR = %.3f)' % average_precision
        )

        # interp_precision = np.interp(mean_recall, recall, precision)
        # # interp_precision[0] = 0.0
        # pres.append(interp_precision)
        # rpc_aucs.append(rpc_auc)

    """CŨ"""
    # y_tests = np.array([])
    # for y_test in arr_true_y:
    #     y_tests = np.hstack((y_tests, y_test.ravel()))
    """MỚI"""
    y_tests = np.array(arr_y_test).ravel()

    # no_skill = len(y_tests[y_tests == 1]) / y_tests.shape[0]
    # ax_rpc.plot(
    #     [0, 1], [no_skill, no_skill],
    #     linestyle='--', lw=2, color='r', label_new='Chance'
    # )

    # Dat ten
    ax_rpc.set_title('Precision-Recall curve')
    ax_rpc.set_xlabel('Recall')
    ax_rpc.set_ylabel('Precision')
    ax_rpc.legend(loc="lower left")

    if save is not None:
        fig.savefig(save + '.eps', format='eps',
                    transparent=True,
                    bbox_inches='tight')
        fig.savefig(save + '.png', format='png',
                    transparent=True,
                    bbox_inches='tight')
    plt.show()
