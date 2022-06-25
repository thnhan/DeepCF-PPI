import numpy as np

from feature_extraction.hand import convert
from feature_extraction.embedding import embedding


def prot2vec(w2v_model, prot_seq_A, prot_seq_B, protlen):
    ###################################################
    # Thêm những phần trích xuất đặc trưng vào đây
    ###################################################
    handcrafted_A = convert(np.array(prot_seq_A))
    handcrafted_B = convert(np.array(prot_seq_B))

    # Lấy đặc trưng bằng Word2vec
    embeddeding_A = embedding(w2v_model, prot_seq_A, protlen)
    embeddeding_B = embedding(w2v_model, prot_seq_B, protlen)

    # Nối đặc trưng đã được trích xuất vào feature_protein_A, B
    protvec_A = np.hstack((embeddeding_A, handcrafted_A))
    protvec_B = np.hstack((embeddeding_B, handcrafted_B))

    # Ghép [A, B]
    protvec_AB = np.hstack((protvec_A, protvec_B))

    return protvec_AB


def prot2embedding(w2v_model, prot_seq_A, prot_seq_B, protlen, handcrafted_A, handcrafted_B):
    # Lấy đặc trưng bằng Word2vec
    embeddeding_A = embedding(w2v_model, prot_seq_A, protlen)
    embeddeding_B = embedding(w2v_model, prot_seq_B, protlen)
    # Nối đặc trưng đã được trích xuất vào feature_protein_A, B
    protvec_A = np.hstack((embeddeding_A, handcrafted_A))
    protvec_B = np.hstack((embeddeding_B, handcrafted_B))
    # Ghép [A, B]
    protvec_AB = np.hstack((protvec_A, protvec_B))
    return protvec_AB
