"""
@author: thnhan
"""
import numpy as np
import pandas as pd


def protein2token(sequences):
    """ @thnhan """
    tokens = []
    for seq in sequences:
        tokens.append(list(seq))
    return tokens


def w2v_transformer(w2v_model, proteins, maxlen):
    def token2vec(token):
        n_dims = len(w2v_model['A'])  # n_dims là số chiều của vector từ
        vec = np.zeros(maxlen * n_dims)
        # print(vec.shape)
        i = 0
        for w in token[:maxlen]:
            temp = w2v_model[w]
            # print(len(temp))
            vec[i:i + len(temp)] = temp
            i += len(temp)
        return vec

    # token
    token_proteins = protein2token(proteins)
    feat_vectors = map(token2vec, token_proteins)

    return np.array(list(feat_vectors))


def embedding(w2v_model, proteins, protlen):
    def token2vec(token):
        vec = lockup_table.loc[token[:protlen]].values
        # ====== padding
        if len(vec) < protlen:
            pad = np.zeros((protlen - len(vec), dim))
            vec = np.vstack((vec, pad))
        return vec

    lockup_table = []
    AA = []  # acid amin == vocal
    for vocal in w2v_model.key_to_index.keys():  # for gensim 4
        # for vocal in list(w2v_model.vocal):  # for gensim 3
        AA.append(vocal)
        lockup_table.append(list(w2v_model[vocal]))
    lockup_table = pd.DataFrame(data=lockup_table, index=AA)

    # token
    dim = len(w2v_model['A'])
    token_proteins = protein2token(proteins)
    feat_vectors = np.zeros((len(token_proteins), protlen * dim))
    for i, token in enumerate(token_proteins):
        vec = token2vec(token).reshape(-1)
        feat_vectors[i, :] = vec
    return feat_vectors
