"""
@thnhan
"""

import re

import numpy as np
import pandas as pd
from sys import path

from feature_extraction.descriptors.AAC import AACEncoder

AA_1 = "ARNDCEQGHILKMFPSTWYV"
AA_idx = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12,
          'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}


def read_supp_table(supp):
    table = pd.read_csv(supp)
    table.set_index('AA', inplace=True)
    for i, col in enumerate(table.columns):
        m = table[col].mean()
        s = table[col].std()
        table[i] = (table[col] - m) / s
    return table[[0, 1, 2]].values


def to_indices(sequence, dict_AA_1):
    # print(sequence)
    return [dict_AA_1[aa] for aa in sequence]


def R_di_peptides():
    proper = read_supp_table(supp_dir + r'/protein_properties.csv')
    R = [[0.0] * 20] * 20
    for ii, aa1 in enumerate(AA_1):
        for jj, aa2 in enumerate(AA_1[ii:]):
            v = (proper[ii] - proper[jj]) ** 2
            v = np.sum(v) / 3
            R[ii][jj] = v
            R[jj][ii] = v
    return R


supp_dir = path[1] + r'/feature_extraction/descriptors'


class PseAACEncoder:
    def __init__(self, remove_unknown_AA, lg=30):
        """ @thnhan """
        self.minLength = lg + 1
        self.dim = 20 + lg
        self.remove_unknown_AA = remove_unknown_AA
        self.shortName = 'PseAAC'
        self.fullName = 'Pseudo-Amino Acid Composition'

    def to_feature(self, sequence, lg=30, omega=0.05):
        # Loại bỏ các axit amin không xác định 'U', 'X'
        if self.remove_unknown_AA:
            sequence = re.sub("[UX]", "", sequence)

        # Thành phần 1
        acc = AACEncoder().to_feature(sequence)

        # Thành phần 2
        R = R_di_peptides()
        seq = to_indices(sequence, AA_idx)
        L = len(sequence)
        theta = [0.0] * lg
        for k in range(1, lg + 1):
            s = sum([R[seq[i]][seq[i + k]] for i in range(L - k)])
            theta[k - 1] = s

        den = 1.0 + omega * sum(theta)
        features = np.hstack((acc, theta))

        return features / den


if __name__ == "__main__":
    sequence = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKAT' \
               'QAAQDGQTTESRKAIQSDIVRLIQGLDNIGNTTTYNGQALLSGQFTNKEFQVGAYSNQSIKASIGSTTSDKIGQVRIATGALITASGDISLTFKQV' \
               'DGVNDVTLESVKVSSSAGTGIGVLAEVINKNSNRTGVKAYASVITTSDVAVQSGSLSNLTLNGIHLGNIADIKKNDSDGRLVAAINAVTSETGVEA' \
               'YTDQKGRLNLRSIDGRGIEIKTDSVSNGPSALTMVNGGQDLTKGSTNYGRLSLTRLDAKSINVVSASDSQHLGFTAIGFGESQVAETTVNLRDVTG' \
               'NFNANVKSASGANYNAVIASGNQSLGSGVTTLRGAMVVIDIAESAMKMLDKVRSDLGSVQNQMISTVNNISITQVNVKAAESQIRDVDFAEESANF' \
               'NKNNILAQSGSYAMSQANTVQQNILRLLT'
    v = PseAACEncoder(remove_unknown_AA=True).to_feature(sequence)
    print(v)
    print(v.shape)
