import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

# tam = [[], [], [], []]
# for i in range(5):
#     # ====== LOAD DATA
#     tr_Xy = h5py.File(r'tr_fold' + str(i) + '.data', 'r')
#     tr_feat_AB, tr_labels = tr_Xy['X'], tr_Xy['y']
#     # print(tr_feat_AB)
#     # print(tr_labels)
#
#     te_Xy = h5py.File(r'te_fold' + str(i) + '.data', 'r')
#     te_feat_AB, te_labels = te_Xy['X'], te_Xy['y']
#
#     rf_model = RandomForestClassifier(n_estimators=500, )
#     rf_model.fit(tr_feat_AB, tr_labels)
#     # print(rf_model.score(te_feat_AB, te_labels))
#     tam[0].append(rf_model.score(te_feat_AB, te_labels))
#
#     xg_model = XGBClassifier(n_estimators=500, learning_rate=0.03)
#     xg_model.fit(tr_feat_AB, tr_labels)
#     # print(xg_model.score(te_feat_AB, te_labels))
#     tam[1].append(xg_model.score(te_feat_AB, te_labels))
#
#
#
#     svm = SVC(C=5., degree=10)
#     svm.fit(tr_feat_AB, tr_labels)
#     # print(svm.score(te_feat_AB, te_labels))
#     tam[3].append(svm.score(te_feat_AB, te_labels))
#
#     tr_Xy.close()
#     te_Xy.close()
#
# # tam = np.array(tam).reshape()
# # print(tam)
# print(np.mean(tam, axis=1))

max_acc = 0.
best_params = dict()

# for C in [0.1, 1, 10, 100, 1000]:
#     for gamma in [1, 0.1, 0.01, 0.001, 0.0001]:
#         for degree in [1, 3, 5, 8]:
#             for kernel in ['rbf', 'poly']:
#                 acc = []
#                 for i in range(5):
#                     # ====== LOAD DATA
#                     tr_Xy = h5py.File(r'tr_fold' + str(i) + '.data', 'r')
#                     tr_feat_AB, tr_labels = tr_Xy['X'], tr_Xy['y']
#                     te_Xy = h5py.File(r'te_fold' + str(i) + '.data', 'r')
#                     te_feat_AB, te_labels = te_Xy['X'], te_Xy['y']
#
#                     lb_model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
#                     lb_model.fit(tr_feat_AB, tr_labels)
#                     acc.append(lb_model.score(te_feat_AB, te_labels))
#
#                     tr_Xy.close()
#                     te_Xy.close()
#
#                 acc = np.mean(acc)
#                 if acc > max_acc:
#                     max_acc = acc
#                     best_params['C'] = C
#                     best_params['Kernel'] = kernel
#                     best_params['Gamma'] = gamma
#                     best_params['Degree'] = degree

# # for trees in [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]:
# for trees in [2, 6, 10]:
#     # for lr in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
#     # for cri in ['gini','entropy']:
#     acc = []
#     for i in range(5):
#         # ====== LOAD DATA
#         tr_Xy = h5py.File(r'tr_fold' + str(i) + '.data', 'r')
#         tr_feat_AB, tr_labels = tr_Xy['X'], tr_Xy['y']
#         te_Xy = h5py.File(r'te_fold' + str(i) + '.data', 'r')
#         te_feat_AB, te_labels = te_Xy['X'], te_Xy['y']
#
#         lb_model = KNeighborsClassifier(n_neighbors=trees) #, criterion=lr)
#         lb_model.fit(tr_feat_AB, tr_labels)
#         acc.append(lb_model.score(te_feat_AB, te_labels))
#
#         tr_Xy.close()
#         te_Xy.close()
#
#     acc = np.mean(acc)
#     if acc > max_acc:
#         max_acc = acc
#         best_params['Number of neighbors'] = trees
#         # best_params['Learning rate'] = lr
#         # best_params['Criterion'] = cri

acc =[]
for i in range(5):
    # ====== LOAD DATA
    tr_Xy = h5py.File(r'tr_fold' + str(i) + '.data', 'r')
    tr_feat_AB, tr_labels = tr_Xy['X'], tr_Xy['y']
    te_Xy = h5py.File(r'te_fold' + str(i) + '.data', 'r')
    te_feat_AB, te_labels = te_Xy['X'], te_Xy['y']

    lb_model = GaussianNB()
    lb_model.fit(tr_feat_AB, tr_labels)
    acc.append(lb_model.score(te_feat_AB, te_labels))

    tr_Xy.close()
    te_Xy.close()

acc = np.mean(acc)
if acc > max_acc:
    max_acc = acc
    # best_params['Number of neighbors'] = trees
    # # best_params['Learning rate'] = lr
    # # best_params['Criterion'] = cri


print(max_acc)
print(best_params)
