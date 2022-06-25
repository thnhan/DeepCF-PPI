def draw_plot_LOOCV(arr_y_test, arr_y_prob, arr_y_pred):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
    ax_roc = ax[0]
    ax_rpc = ax[1]

    # get fpr, tpr scores
    fpr, tpr, _ = roc_curve(arr_y_test, arr_y_prob)
    roc_auc = auc(fpr, tpr)
    # plot ROC curve
    ax_roc.plot(fpr, tpr, lw=2, alpha=0.8, label='LOOCV ROC (AUC = %0.2f)'% (roc_auc))
    ax_roc.plot([0, 1], [0, 1], lw=2, alpha=.8, linestyle='--', color='r', label='Chance')
    ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic - SVM")
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")

    # get precision, recall scores
    precision, recall,  _ = precision_recall_curve(arr_y_test, arr_y_prob)
    rpc_auc = average_precision_score(arr_y_test, arr_y_prob)
    # plot precision recall curve
    ax_rpc.plot(recall, precision, lw=2, alpha=0.8, label='LOOCV PR (AP = %0.2f)'% (rpc_auc))
    no_skill = len(arr_y_test[arr_y_test==1]) / arr_y_test.shape[0]
    ax_rpc.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='r', label='Chance')
    ax_rpc.set_title('Recall Precision Curve - SVM')
    ax_rpc.set_xlabel('Recall')
    ax_rpc.set_ylabel('Precision')
    ax_rpc.legend(loc="lower left")
    
    plt.savefig('./results/LOOCV/AC_plot_SVM.png', format='png')
    plt.savefig('./results/LOOCV/AC_plot_SVM.pdf', format='pdf')
    plt.show