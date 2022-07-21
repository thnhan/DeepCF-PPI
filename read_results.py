import pickle

import pandas as pd


def read_prediction_on_test_set(dset_name):
    y_prob = pickle.load(open('predictions_' + dset_name + '.pkl', 'rb'))
    y_prob = pd.DataFrame(y_prob, columns=['Class 0', 'Class 1'])
    print(dset_name, "- Classification probability\n", y_prob)
    return y_prob


if __name__ == "__main__":
    y_prob = read_prediction_on_test_set('Cancer_specific')
    read_prediction_on_test_set('Ecoli')