"""
@author: thnhan
"""

import pandas as pd
import numpy as np

from datasets.fasta_tool import fasta_to_dataframe


def get_4_vectors(X, embedding_dim, handfeat_dim):
    tam = 0
    trai_doan_word2vec = X[:, tam:tam + embedding_dim]
    tam += embedding_dim
    # print(tam)
    trai_doan_seq_feat = X[:, tam:tam + handfeat_dim]
    tam += handfeat_dim
    # print(tam)
    phai_doan_word2vec = X[:, tam:tam + embedding_dim]
    tam += embedding_dim
    # print(tam)
    phai_doan_seq_feat = X[:, tam:tam + handfeat_dim]
    # print(tam + handfeat_dim)
    return trai_doan_word2vec, trai_doan_seq_feat, phai_doan_word2vec, phai_doan_seq_feat


def load_raw_dset(dset_dir):
    seq = pd.read_csv(dset_dir + '/uniprotein.txt', index_col=0, sep="\t")
    pos = pd.read_csv(dset_dir + '/positive.txt', sep="\t")
    neg = pd.read_csv(dset_dir + '/negative.txt', sep="\t")

    do_dai = sorted([len(p) for p in seq.protein])
    avelen = sum(do_dai) / len(do_dai)
    summary = {'minlen': do_dai[0], 'maxlen': do_dai[-1], 'avelen': avelen, 'n_proteins': len(do_dai)}

    # print("TB cua 95% lon nhat", np.mean(do_dai[int(0.05 * len(do_dai)):]))
    # print("TB cua 90% lon nhat", np.mean(do_dai[int(0.1 * len(do_dai)):]))
    # print("TB cua 80% lon nhat", np.mean(do_dai[int(0.2 * len(do_dai)):]))
    # print("TB cua 60% lon nhat", np.mean(do_dai[int(0.4 * len(do_dai)):]))
    # print("TB cua 40% lon nhat", np.mean(do_dai[int(0.6 * len(do_dai)):]))

    P_seq_A = seq.loc[pos.proteinA]['protein'].values
    P_seq_B = seq.loc[pos.proteinB]['protein'].values
    N_seq_A = seq.loc[neg.proteinA]['protein'].values
    N_seq_B = seq.loc[neg.proteinB]['protein'].values

    labels = np.array([1] * len(pos) + [0] * len(neg))
    pairs = np.vstack((pos.values, neg.values))
    dset = {"labels": labels, "id_pairs": pairs, "seq_pairs": (P_seq_A, P_seq_B, N_seq_A, N_seq_B)}
    return dset, summary


def load_Yeastfull_new(dset_dir):
    seq = fasta_to_dataframe(dset_dir + '/uniprotein.txt')
    P_seq_A = fasta_to_dataframe(dset_dir + '/pos_A.txt')
    P_seq_B = fasta_to_dataframe(dset_dir + '/pos_B.txt')
    N_seq_A = fasta_to_dataframe(dset_dir + '/neg_A.txt')
    N_seq_B = fasta_to_dataframe(dset_dir + '/neg_B.txt')

    # P_seq_A = P_seq_A['protein'].values
    # P_seq_B = P_seq_B['protein'].values
    # N_seq_A = N_seq_A['protein'].values
    # N_seq_B = N_seq_B['protein'].values

    do_dai = [len(p) for p in seq.protein]
    avelen = sum(do_dai) / len(do_dai)
    summary = {'minlen': do_dai[0], 'maxlen': do_dai[-1], 'avelen': avelen, 'n_proteins': len(do_dai)}

    yeastfull = pd.read_csv(dset_dir + "/yeastfull_pair.txt")
    labels = yeastfull['interaction'].values
    pairs = yeastfull[['proteinA', 'proteinB']]
    dset = {"labels": labels, "id_pairs": pairs, "uniprot": seq, "seq_pairs": (P_seq_A, P_seq_B, N_seq_A, N_seq_B)}
    return dset, summary


def load_Yeastfull(dset_dir):
    seq = fasta_to_dataframe(dset_dir + '/uniprotein.txt')
    P_seq_A = fasta_to_dataframe(dset_dir + '/pos_A.txt')
    P_seq_B = fasta_to_dataframe(dset_dir + '/pos_B.txt')
    N_seq_A = fasta_to_dataframe(dset_dir + '/neg_A.txt')
    N_seq_B = fasta_to_dataframe(dset_dir + '/neg_B.txt')

    P_seq_A = P_seq_A['protein'].values
    P_seq_B = P_seq_B['protein'].values
    N_seq_A = N_seq_A['protein'].values
    N_seq_B = N_seq_B['protein'].values

    do_dai = sorted([len(p) for p in seq.protein])
    avelen = sum(do_dai) / len(do_dai)
    summary = {'minlen': do_dai[0], 'maxlen': do_dai[-1], 'avelen': avelen, 'n_proteins': len(do_dai)}

    yeastfull = pd.read_csv(dset_dir + "/yeastfull_pair.txt")
    labels = yeastfull['interaction'].values
    pairs = yeastfull[['proteinA', 'proteinB']].values
    dset = {"labels": labels, "id_pairs": pairs, "seq_pairs": (P_seq_A, P_seq_B, N_seq_A, N_seq_B)}
    return dset, summary


if __name__ == "__main__":
    dset, summary = load_raw_dset(r"./Human8161")
    print("summary:", summary)
    print(dset.keys())
    P_seq_A, P_seq_B, N_seq_A, N_seq_B = dset['seq_pairs']
    print("pair\n", dset['pair'])
    print("label\n", dset['label'])
    print("P_seq_A\n", P_seq_A)

    dset, summary = load_Yeastfull(r"YeastFull")
    print("summary:", summary)
    print(dset.keys())
    P_seq_A, P_seq_B, N_seq_A, N_seq_B = dset['seq']
    print("pair\n", dset['pair'])
    print("label\n", dset['label'])
    print("P_seq_A\n", P_seq_A)
