"""
http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/type2.htm

Type 2 PseAA composition is also called the series-correlation type and generates 20 + i*lambda
discrete numbers to represent a protein (i is the number of amino acid attributes selected),
which was introduced by Prof. Kuo-Chen Chou in 2005 and the related publications are:

(1) Chou, K.C. (2005). Using amphiphilic pseudo amino acid composition to predict enzyme subfamily classes,
Bioinformatics, 21, 10-19.
(2) Chou,K.C. and Cai Y.D. (2005). Prediction of membrane protein types by incorporating amphipathic effects,
J Chem Inf Model, 45(2):407-13
"""

import numpy as np
import pandas as pd
from sys import path

from feature_extraction.descriptors.AAC import AACEncoder

AA_1 = "ARNDCEQGHILKMFPSTWYV"
AA_idx = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12,
          'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}


def read_supp_table():
    # print(supp_filename)
    table = pd.read_csv(supp_path)
    table.set_index('AA', inplace=True)
    return table


def to_indices(protseq, dict_AA_1):
    return [dict_AA_1[aa] for aa in protseq]


tau1 = [[0.0] * 20] * 20
tau2 = [[0.0] * 20] * 20


def tau_di_peptides():
    supp_table = read_supp_table()
    proper1 = supp_table['H1'].tolist()
    proper2 = supp_table['H2'].tolist()
    global tau1, tau2
    for i, aa1 in enumerate(AA_1):
        for j, aa2 in enumerate(AA_1[i:]):
            v = proper1[i] * proper1[j]
            tau1[i][j] = v
            tau1[j][i] = v
            v = proper2[i] * proper2[j]
            tau2[i][j] = v
            tau2[j][i] = v


supp_path = path[1] + r"/feature_extraction/descriptors/supp_APAAC.csv"


class APAACEncoder:
    def __init__(self, remove_unknown_AA, lg=30, supp=None):
        if supp is not None:
            self.supp_table = read_supp_table()
        else:
            self.supp_table = read_supp_table()
        self.remove_unknown_AA = remove_unknown_AA
        self.minLength = 31
        self.dim = 20 + 2 * lg
        self.shortName = 'APAAC'
        self.fullName = 'Amphiphilic Pseudo Amino Acid Composition'

    def to_feature(self, protseq, lg=30, omega=0.5):
        if self.remove_unknown_AA:
            protseq = protseq.replace('U', '')
            protseq = protseq.replace('X', '')

        # Thành phần 1
        F = AACEncoder().to_feature(protseq)

        # Thành phần 2
        tau_di_peptides()
        seq = to_indices(protseq, AA_idx)
        tau = [0.0] * (2 * lg)
        L = len(protseq)
        i = 0
        for k in range(1, lg + 1):
            s1 = sum([tau1[seq[i]][seq[i + k]] for i in range(L - k)])
            s2 = sum([tau2[seq[i]][seq[i + k]] for i in range(L - k)])
            tau[i] = s1
            tau[i + 1] = s2
            i += 2

        mau = 1.0 + omega * sum(tau)
        features = np.hstack((F, tau))

        return features / mau


if __name__ == "__main__":
    sequence = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKAT' \
               'QAAQDGQTTESRKAIQSDIVRLIQGLDNIGNTTTYNGQALLSGQFTNKEFQVGAYSNQSIKASIGSTTSDKIGQVRIATGALITASGDISLTFKQV' \
               'DGVNDVTLESVKVSSSAGTGIGVLAEVINKNSNRTGVKAYASVITTSDVAVQSGSLSNLTLNGIHLGNIADIKKNDSDGRLVAAINAVTSETGVEA' \
               'YTDQKGRLNLRSIDGRGIEIKTDSVSNGPSALTMVNGGQDLTKGSTNYGRLSLTRLDAKSINVVSASDSQHLGFTAIGFGESQVAETTVNLRDVTG' \
               'NFNANVKSASGANYNAVIASGNQSLGSGVTTLRGAMVVIDIAESAMKMLDKVRSDLGSVQNQMISTVNNISITQVNVKAAESQIRDVDFAEESANF' \
               'NKNNILAQSGSYAMSQANTVQQNILRLLT'
    v = APAACEncoder(remove_unknown_AA=True).to_feature(sequence)
    print(v)
    print(v.shape)
