"""
Performance on Yeast core datasets:
1. Using 5-fold cross-validation on "Human8161" datasets.
2. Params selection

@author: thnhan
"""
import copy
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from datasets.dset_tool import load_raw_dset, get_4_vectors
# from handfeat.load_handfeat import load_handfeat_Human8161
from feature_extraction.protein2vector import prot2vec
from models.dnn_attention import dnn_att_model
from models.dnn_model import net
from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import print_metrics, my_cv_report


def get_avelen(inds):
    pos_A, pos_B, neg_A, neg_B = dset['seq_pairs']
    pos_AB = np.hstack((pos_A, pos_B))
    neg_AB = np.hstack((neg_A, neg_B))
    prots = np.concatenate((pos_AB, neg_AB), axis=0)
    prots = prots.flatten()
    prots = prots[inds]
    prots = np.unique(prots)
    do_dai = [len(seq) for seq in prots]
    avelen = int(sum(do_dai) / len(do_dai))
    return avelen


def prepare_Human8161_feat(w2v_model, protlen):
    pos_seq_A, pos_seq_B, neg_seq_A, neg_seq_B = dset['seq_pairs']
    pos_feature_AB = prot2vec(w2v_model, pos_seq_A, pos_seq_B, protlen)
    neg_feature_AB = prot2vec(w2v_model, neg_seq_A, neg_seq_B, protlen)
    feature_AB = np.vstack((pos_feature_AB, neg_feature_AB))
    feature_AB = StandardScaler().fit_transform(feature_AB)
    return copy.deepcopy(feature_AB)


def eval_model(pairs, labels):
    start_time = time.time()

    skf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
    scores = []
    hists = []
    cv_prob_y, cv_test_y = [], []

    method_result = dict()

    for i, (tr_inds, te_inds) in enumerate(skf.split(pairs, labels)):
        print("\nFold", i)
        protlen = get_avelen(tr_inds)
        print("Average length:", protlen)
        feat = prepare_Human8161_feat(trained_w2v.wv, protlen)

        Y = to_categorical(labels)
        tr_X, te_X = feat[tr_inds], feat[te_inds]
        tr_Y, te_Y = Y[tr_inds], Y[te_inds]

        # ====== get 4 handfeat
        tr_A_w2v, tr_A_seq, tr_B_w2v, tr_B_seq = get_4_vectors(tr_X, protlen * AAsize, handdim)
        te_A_w2v, te_A_seq, te_B_w2v, te_B_seq = get_4_vectors(te_X, protlen * AAsize, handdim)

        # ====== DEF MODEL
        if os.path.exists('ATT_OurModel_trained_on_Human8161_fold' + str(i) + '.h5'):
            model = load_model('ATT_OurModel_trained_on_Human8161_fold' + str(i) + '.h5')
            print("====== LOADED MODEL")
        else:
            # model = net(protlen * AAsize, handdim, None, n_units=1024)
            model = dnn_att_model(protlen * AAsize, handdim, None, n_units=1024)  # with attention

            opt = Adam(decay=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            # ====== FIT MODEL
            hist = model.fit([tr_A_w2v, tr_A_seq, tr_B_w2v, tr_B_seq], tr_Y,
                             batch_size=256,  # 256
                             epochs=45,  # 45,
                             verbose=0)
            hists.append(hist)

            # ====== SAVE MODEL
            model.save("ATT_OurModel_trained_on_Human8161_fold" + str(i) + ".h5")

        # ====== REPORT
        prob_Y = model.predict([te_A_w2v, te_A_seq, te_B_w2v, te_B_seq])

        # ====== Keep for comparing with methods, thnhan
        method_result['fold' + str(i)] = {"true_y": np.argmax(te_Y, axis=1),
                                          "prob_Y": prob_Y}
        pickle.dump(method_result, open(r'ATT_trained_5CV_predictions_on_Human8161.pkl', 'wb'))
        # ======

        scr = print_metrics(np.argmax(te_Y, axis=1), prob_Y)
        scores.append(scr)

        cv_prob_y.append(prob_Y[:, 1])
        cv_test_y.append(np.argmax(te_Y, axis=1))

    # ====== FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)

    print("Running time", time.time() - start_time)

    # plot_folds(plt, cv_test_y, cv_prob_y)
    # plt.show()
    return hists


# ====== GLOBAL HYPER PARAMETERS
AAsize = 20  # word (amino acid) size
handdim = 650  # handcrafted feature dimension
trained_w2v = Word2Vec.load(r"feature_extraction/w2v_embedding/trained_AAsize20.wv")

if __name__ == "__main__":
    dset, summary = load_raw_dset("datasets/Human8161")
    id_pairs = dset['id_pairs']
    labels = dset['labels']
    print("Summary:", summary)
    print("Number of pairs:", len(id_pairs))
    eval_model(id_pairs, labels)
