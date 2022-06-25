"""
@author: thnhan
"""

import pandas as pd


def get_protein_from_fasta(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        inds = []
        if len(lines) > 0:
            for i, l in enumerate(lines):
                if l.startswith('>'):
                    inds.append(i)
            inds.append(len(lines))

            sequences = []
            for i in range(len(inds) - 1):
                seq = lines[inds[i]:inds[i + 1]]
                seq = ''.join(seq[1:])
                seq = seq.replace('\n', '')
                sequences.append(seq)
        else:
            print("====== FILE is EMPTY =======")
    return sequences


def fasta_to_dataframe(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        inds = []
        if len(lines) > 0:
            for i, l in enumerate(lines):
                if l.startswith('>'):
                    inds.append(i)
            inds.append(len(lines))

            sequences = []
            IDs = []
            for i in range(len(inds) - 1):
                item = lines[inds[i]:inds[i + 1]]
                IDs.append(item[0].replace(">", "").strip("\n"))
                seq = ''.join(item[1:])
                seq = seq.replace('\n', '')
                sequences.append(seq)
        else:
            print("====== FILE is EMPTY =======")
    data = pd.DataFrame(data=sequences, index=IDs, columns=['protein'])
    return data
