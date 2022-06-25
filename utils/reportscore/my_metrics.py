"""
Tính các loại điểm nhằm đánh giá năng lực của mô hình
@author: thnhan
"""
from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, auc, roc_curve
# from src.my_tools.utils.tools import calculate_performance


# def performance_report(model, test_X, test_y):
#     """ @author: thnhan """
#     from sklearn.metrics import (plot_confusion_matrix,
#                                  plot_precision_recall_curve,
#                                  plot_roc_curve)
#     from matplotlib import cm
#
#     pred_y = model.predict(test_X)
#     # print(classification_report(test_y, pred_y))
#     # print('ACC on test: %.3f' % accuracy_score(test_y, pred_y))
#
#     # Report scores
#     scores = calculate_performance(test_y, pred_y)
#     print('Accuracy:    %.2f\n'
#           'Sensitivity: %.2f\n'
#           'Precision:   %.2f\n'
#           'F1-score:    %.2f\n'
#           'MCC-score:   %.2f' % (scores['Accuracy'] * 100,
#                                  scores['Sensitivity'] * 100,
#                                  scores['Precision'] * 100,
#                                  scores['F1-score'] * 100,
#                                  scores['MCC-score'] * 100))
#
#     # Plot
#     plot_confusion_matrix(model, test_X, test_y,
#                           normalize='true',
#                           cmap=cm.cmap_d['Blues'])
#     plot_precision_recall_curve(model, test_X, test_y)
#     plot_roc_curve(model, test_X, test_y)


if __name__ == "__main__":
    print(calculate_scores([1, 1, 0, 1, 1], [1, 0, 0, 1, 0]))
