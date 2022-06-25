""" @coding: thnhan """
from sys import path

import numpy as np
import pandas as pd
from feature_extraction.descriptors.AAC import AACEncoder

AA_idx = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12,
          'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}


def to_indices(protseq, dict_AA_1):
    return [dict_AA_1[aa] for aa in protseq]


def tau_di_peptides():
    delta1 = pd.read_csv(supp_dir + r'\Grantham.csv',
                         index_col=0, sep='\t').values  # Grantham
    delta2 = pd.read_csv(supp_dir + r'\Schneider-Wrede.csv',
                         index_col=0, sep='\t').values  # Schneider-Wrede
    return delta1 * delta1, delta2 * delta2


supp_dir = path[1] + r'/feature_extraction/descriptors'


class QSOEncoder:
    def __init__(self, remove_unknown_AA, lg=30):
        self.remove_unknown_AA = remove_unknown_AA
        self.minLength = lg + 1
        self.dim = 40 + 2 * lg
        self.shortName = 'QSO'
        self.fullName = 'Quasi Sequence Order'

    def to_feature(self, protseq, lg=30, omega=0.1):
        if self.remove_unknown_AA:
            protseq = protseq.replace('U', '')
            protseq = protseq.replace('X', '')

        # Thành phần 1
        F = AACEncoder().to_feature(protseq)

        # Thành phần 2
        delta1, delta2 = tau_di_peptides()
        seq = to_indices(protseq, AA_idx)

        # print(seq)
        tau1 = np.zeros(lg)
        tau2 = np.zeros(lg)
        L = len(protseq)
        for k in range(lg):
            s1 = sum([delta1[seq[i]][seq[i + k]] for i in range(L - k - 1)])
            s2 = sum([delta2[seq[i]][seq[i + k]] for i in range(L - k - 1)])
            tau1[k - 1] = s1
            tau2[k - 1] = s2

        mau1 = 1.0 + omega * sum(tau1)
        mau2 = 1.0 + omega * sum(tau2)
        features = np.hstack((F / mau1, F / mau2, tau1 / mau1, tau2 / mau2))

        return features
