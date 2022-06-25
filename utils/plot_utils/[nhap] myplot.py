import numpy as np
from sklearn.metrics import plot_roc_curve, precision_recall_curve, average_precision_score
from scipy import interp

def draw_cv_roc_curve(ax, estimator, n_samples, X_fold, y_fold, i):
    mean_fpr = np.linspace(0, 1, n_samples)
    # Ve bieu do Moi them vao.  {...
    viz = plot_roc_curve(estimator, X_fold, y_fold,
                        name='ROC fold {}'.format(i),
                        alpha=0.3, lw=1, ax=ax)
    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    # tprs.append(interp_tpr)

    return interp_tpr


    y_test = np.concatenate(y_test)
    y_proba = estimator.predict_proba(X_test) # np.concatenate(y_proba)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    axes.plot(recall, precision, color='b',
             label=r'Overall Precision-Recall (AP = %0.3f)' % 
             (average_precision_score(y_test, y_proba)),
             lw=2, alpha=.8)
            

