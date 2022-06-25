# -*- coding: utf-8 -*-
"""
@author: thnhan
"""
import os.path
from sys import path
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from gensim.models.word2vec import Word2Vec
from tensorflow.python.keras.utils.np_utils import to_categorical
import h5py
from datasets.dset_tool import get_4_vectors, load_raw_dset
from datasets.fasta_tool import get_protein_from_fasta
from feature_extraction.protein2vector import prot2vec
from models.dnn_attention import dnn_att_model
from models.dnn_model import net
from perform_on_YeastCore import prepare_YeastCore_feat, get_avelen


# def train_model(X, y):
#     tr_model = net(protlen * AAsize, hcfdim, None, n_units=1024)
#     opt = Adam(decay=0.001)
#     tr_model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])
#     # ====== FIT MODEL
#     tr_model.fit(X, y,
#                  batch_size=64,
#                  epochs=epochs,  # 32,
#                  verbose=1)
#     tr_model.save(model_filename)  # thnhan
#     print('\n====== SAVED {}.'.format(model_filename))
#     return tr_model


def load_trained_model():
    trained_model = load_model(model_filename)
    print('====== Loaded Model .')
    return trained_model


def prepare_independent_dataset(w2v_model, protfile_A, protfile_B, protlen):
    # Get sequences
    protseq_A = get_protein_from_fasta(protfile_A)
    protseq_B = get_protein_from_fasta(protfile_B)
    protvec_AB = prot2vec(w2v_model, protseq_A, protseq_B, protlen)  # convert to feature vector
    labels = np.ones(len(protvec_AB), dtype=int)  # labels
    return protvec_AB, labels


def run_test(file_prot_A, file_prot_B):
    te_feat_AB, labels = prepare_independent_dataset(trained_w2v.wv,
                                                     file_prot_A,
                                                     file_prot_B,
                                                     protlen)  # label = 1
    te_feat_AB = np.array(te_feat_AB)

    if scal is not None:
        te_feat_AB = scal.transform(te_feat_AB)

    A_w2v, A_seq, B_w2v, B_seq = get_4_vectors(te_feat_AB, protlen * AAsize, hcfdim)

    y_prob = tr_model.predict([A_w2v, A_seq, B_w2v, B_seq])
    y_pred = np.argmax(y_prob, axis=1).astype(int)
    ACC = sum(y_pred == labels) / len(y_pred)
    print('> Accuracy {}'.format(ACC))
    return ACC


def test_on_species(dataset_name):
    print('\n====== Testing on {} ...'.format(dataset_name))
    file_prot_A = species_datadir + r'/' + dataset_name + r'_ProA.txt'
    file_prot_B = species_datadir + r'/' + dataset_name + r'_ProB.txt'
    return run_test(file_prot_A, file_prot_B)


def test_on_network_data(dataset_name):
    print('\n====== Testing on {} ...'.format(dataset_name))
    file_prot_A = path[0] + r'/' + network_datadir + r'/' + dataset_name + r'/' + dataset_name + r'_ProA.txt'
    file_prot_B = path[0] + r'/' + network_datadir + r'/' + dataset_name + r'/' + dataset_name + r'_ProB.txt'
    return run_test(file_prot_A, file_prot_B)


# ====== GLOBAL HYPER PARAMETERS
AAsize = 20  # word (amino acid) size
hcfdim = 650  # handcrafted feature dimension
epochs = 32
w2v_filename = r'feature_extraction/w2v_embedding/trained_AAsize20.wv'
species_datadir = r'datasets/Cross_species'
network_datadir = r'datasets/Networks'
loss_fn = 'categorical_crossentropy'
model_filename = 'LAN_CHAY/att, 5 neurons/ATT_OurModel_trained_on_Yeastcore.h5'
results_filename = 'Results_of_test.txt'

if __name__ == "__main__":
    trained_w2v = Word2Vec.load(w2v_filename)
    dset, summary = load_raw_dset("datasets/Yeastcore")
    id_pairs = dset['id_pairs']
    tr_labels = dset['labels']
    print("Summary:", summary)
    print("Number of pairs:", len(id_pairs))
    inds = np.arange(len(id_pairs))
    protlen = get_avelen(inds, dset)
    print("Average length:", protlen)

    if not os.path.exists('Yeastcore_AAsize20.data'):
        # ====== PREPARE DATA
        tr_feat_AB = prepare_YeastCore_feat(trained_w2v.wv, protlen, dset)

        # ====== SAVE DATA
        h5_file = h5py.File('Yeastcore_AAsize20.data', 'w')
        h5_file.create_dataset('X', data=tr_feat_AB)
        h5_file.create_dataset('y', data=tr_labels)
        h5_file.close()

    # ====== LOAD DATA
    h5_Xy = h5py.File(r'Yeastcore_AAsize20.data', 'r')
    tr_feat_AB = h5_Xy['X']
    print(tr_feat_AB)
    tr_labels = h5_Xy['y']
    print(tr_labels)

    scal = StandardScaler().fit(tr_feat_AB)
    tr_feat_AB = scal.transform(tr_feat_AB)

    if os.path.exists(model_filename):
        tr_model = load_trained_model()
    else:
        print("\nTrain model...")
        tr_w2v_A, tr_seq_A, tr_w2v_B, tr_seq_B = get_4_vectors(tr_feat_AB, protlen * AAsize, hcfdim)
        # tr_model = net(protlen * AAsize, hcfdim, None, n_units=1024)
        tr_model = dnn_att_model(protlen * AAsize, hcfdim, None, n_units=1024)  # with attention

        opt = Adam(decay=0.001)
        tr_model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])

        # ====== FIT MODEL
        tr_labels = to_categorical(tr_labels)
        tr_model.fit([tr_w2v_A, tr_seq_A, tr_w2v_B, tr_seq_B], tr_labels,
                     batch_size=64,
                     epochs=epochs,  # 32,
                     verbose=1)

        tr_model.save(model_filename)  # thnhan
        print('\n====== SAVED {}.'.format(model_filename))

    # ====== TEST
    trained_w2v = Word2Vec.load(w2v_filename)
    toan_bo_ket_qua = dict()
    toan_bo_ket_qua.update({'Cancer_specific': test_on_network_data('Cancer_specific')})
    toan_bo_ket_qua.update({'One_core': test_on_network_data('One_core')})
    toan_bo_ket_qua.update({'Wnt_related': test_on_network_data('Wnt_related')})
    toan_bo_ket_qua.update({'Celeg': test_on_species('Celeg')})
    toan_bo_ket_qua.update({'Ecoli': test_on_species('Ecoli')})
    toan_bo_ket_qua.update({'Hpylo': test_on_species('Hpylo')})
    toan_bo_ket_qua.update({'Hsapi': test_on_species('Hsapi')})
    toan_bo_ket_qua.update({'Mmusc': test_on_species('Mmusc')})
    toan_bo_ket_qua.update({'Dmela': test_on_species('Dmela')})

    # ====== Lưu kết quả vào file
    with open(results_filename, 'w') as f:
        for d, acc in toan_bo_ket_qua.items():
            f.write(d + "\t" + str(acc) + "\n")
    h5_Xy.close()
