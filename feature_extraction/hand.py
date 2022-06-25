import pandas as pd
import numpy as np

from feature_extraction.descriptors.AmphiphilicPAAC import APAACEncoder
from feature_extraction.descriptors.AAC import AACEncoder
from feature_extraction.descriptors.DPC import DPCEncoder
from feature_extraction.descriptors.PseAAC import PseAACEncoder
from feature_extraction.descriptors.QSO import QSOEncoder


def prot_to_feat(protein, all_enc, remove_unknown_AA=False):
    # Loại bỏ các axit amin không xác định 'U', 'X'
    if remove_unknown_AA:
        # protein = re.sub("[UX]", "Z", protein)
        protein = protein.replace("U", "")
        protein = protein.replace("X", "")

    feat = [enc.to_feature(protein) for enc in all_enc]
    # handfeat = np.concatenate(handfeat)
    # f1 = all_enc[0].to_feature(protein)
    # f2 = all_enc[1].to_feature(protein)
    return np.concatenate(feat)


def prot_list_to_feat(proteins, all_enc, remove_unknown_AA=False):
    """ @thnhan
    1. Lọc ra các trình tự protein khác nhau
    2. Chuyển trình tự protein thành vector đặc trưng
    3. Lắp ghếp các vector đặc trưng theo đúng thứ tự các chuỗi trong proteins
    """

    if len(proteins) >= 500:
        uni_prots = pd.unique(proteins)  # uni_prots chứa những protein duy nhất
        uni_feats = map(lambda prot: prot_to_feat(prot, all_enc, remove_unknown_AA), uni_prots)
        # print(list(uni_feats))
        feat_list = pd.DataFrame(data=list(uni_feats), index=uni_prots)
        feat_list = feat_list.loc[proteins]
        return feat_list.values
    else:
        feat_list = map(lambda prot: prot_to_feat(prot, all_enc), proteins)
        return np.array(list(feat_list))


def convert(proteins):
    protein_encoders = [
        AACEncoder(),
        PseAACEncoder(remove_unknown_AA=True),
        APAACEncoder(remove_unknown_AA=True),
        QSOEncoder(remove_unknown_AA=True),
        DPCEncoder(),
    ]

    # print("Number of proteins:", len(proteins))
    all_feat = prot_list_to_feat(proteins,
                                 protein_encoders,
                                 remove_unknown_AA=True)
    # print("Done.")
    return all_feat
