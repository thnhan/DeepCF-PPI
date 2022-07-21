import numpy as np

from feature_extraction.hand import convert


def FASTA_feat_to_numpy(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        inds = []
        if len(lines) > 0:
            for i, l in enumerate(lines):
                if l.startswith('>'):
                    inds.append(i)
            inds.append(len(lines))
            feat = []
            for i in range(len(inds) - 1):
                item = lines[inds[i]:inds[i + 1]]
                a = ''.join(item[1:]).replace('\n', '')
                a = a.strip("\n").split(",")
                feat.append([float(temp) for temp in a])
        else:
            print("====== FILE is EMPTY =======")
    return np.array(feat)


def FASTA_to_ID_SEQ(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        inds = []
        if len(lines) > 0:
            for i, l in enumerate(lines):
                if l.startswith('>'):
                    inds.append(i)
            inds.append(len(lines))
            IDs, SEQs = [], []
            for i in range(len(inds) - 1):
                item = lines[inds[i]:inds[i + 1]]
                IDs.append(item[0].replace(">", "").strip("\n"))
                seq = ''.join(item[1:])
                seq = seq.replace('\n', '')
                SEQs.append(seq)
        else:
            print("====== FILE is EMPTY =======")
    return IDs, SEQs


file_handfeat = ["feat_pos_A.txt", "feat_pos_B.txt", "feat_neg_A.txt", "feat_neg_B.txt"]
file_sequence = ["pos_A.txt", "pos_B.txt", "neg_A.txt", "neg_B.txt"]
for f1, f2 in zip(file_handfeat, file_sequence):
    # ====== Load handcrafted feature from file
    feat1 = FASTA_feat_to_numpy("handfeat/YeastFull" + "/" + f1)

    # ====== Convert proteins to handcrafted feature
    _, SEQs_A = FASTA_to_ID_SEQ("datasets/YeastFull" + "/" + f2)
    # print(len(IDs_A), len(SEQs_A))
    feat2 = convert(np.array(SEQs_A))

    print("Is close:", np.all(np.isclose(feat1, feat2)))
