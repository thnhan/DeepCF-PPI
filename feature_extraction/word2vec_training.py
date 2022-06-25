"""
Word2vec training module
1. Using "uniprot_sprot.fasta" as input data.
2. Transforming protein sequence to token sequence. Output is a copus.
3. Train Word2vec using skip-gram algorithm.

@author: thnhan
ref:
https://radimrehurek.com/gensim/models/word2vec.html
https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.Yg5UxvVByUk
"""

import gensim
from sys import path
import numpy as np


def train_word2vec(corpus, **kw):
    size = kw['embsize']
    min_count = kw['min_count']
    maxiter = kw['maxiter']
    model = gensim.models.Word2Vec(sentences=corpus,
                                   size=size,
                                   window=5,  # default
                                   min_count=min_count,
                                   workers=4,
                                   iter=maxiter,  # default
                                   seed=1,  # default
                                   sg=1)
    model.save("trained_AAsize{}.wv".format(size))
    return model


def protein2token(proteins):
    tokens = []
    for prot in proteins:
        tokens.append(list(prot))
    return tokens


def lay_chuoi_tu_FASTA(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        inds = []
        if len(lines) > 0:
            for i, l in enumerate(lines):
                if l.startswith('>'):
                    inds.append(i)
            inds.append(len(lines))

            all_seq = []
            for i in range(len(inds) - 1):
                seq = lines[inds[i]:inds[i + 1]]
                # print(seq)
                seq = ''.join(seq[1:])
                seq = seq.replace('\n', '')
                all_seq.append(seq)
        else:
            print("====== FILE is EMPTY =======")

    return all_seq


embsizes = [8, 16, 20, 24, 32]
maxiter = 5

if __name__ == "__main__":
    sequences = lay_chuoi_tu_FASTA(path[1] + r'\datasets\uniprot_sprot.fasta')
    print("====== Number of protein sequences: ", len(sequences))

    uniprot_token = protein2token(sequences)  # 1-gram
    for embsize in embsizes:
        params = {'embsize': embsize,
                  'maxiter': maxiter,
                  'min_count': 0}
        train_word2vec(uniprot_token, **params)
