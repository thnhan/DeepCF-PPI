"""
@thnhan
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, add, Concatenate, Lambda
from tensorflow.keras.layers import Reshape, AveragePooling1D, GlobalAveragePooling1D, Flatten, Conv1D, GlobalMaxPool1D
from tensorflow.keras.initializers import GlorotUniform

import tensorflow as tf


def module_feature_extraction(n_dim, W_regular, drop, n_units, kernel_init):
    dnn = Sequential()
    dnn.add(Dense(n_units, input_dim=n_dim,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 2,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 4,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 8,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    return dnn


def tich(x):
    # print(x[0])
    # print(x[1])
    # a = x[1][0]
    # print(a)

    return x[0] * x[1]


def dnn_att_model(dim1, dim2, W_regular, drop=0.5, n_units=1024, seed=123456):
    """
    - dim1 = fixed protein length * word size, i.e. 557 * 20 = 11140
    - dim2 = 650
    """
    # ====== To reproduce
    tf.random.set_seed(seed)
    glouni = GlorotUniform(seed=seed)

    # ====== Extraction
    w1 = module_feature_extraction(dim1,
                                   W_regular=W_regular,
                                   drop=drop,
                                   n_units=n_units,
                                   kernel_init=glouni)
    s1 = module_feature_extraction(dim2,
                                   W_regular=W_regular,
                                   drop=drop,
                                   n_units=n_units,
                                   kernel_init=glouni)
    w2 = module_feature_extraction(dim1,
                                   W_regular=W_regular,
                                   drop=drop,
                                   n_units=n_units,
                                   kernel_init=glouni)
    s2 = module_feature_extraction(dim2,
                                   W_regular=W_regular,
                                   drop=drop,
                                   n_units=n_units,
                                   kernel_init=glouni)

    in1, in2 = Input(dim1), Input(dim2)
    in3, in4 = Input(dim1), Input(dim2)
    x1, x2 = w1(in1), s1(in2)
    x3, x4 = w2(in3), s2(in4)

    # ====== Attention 12
    x12 = Concatenate()([x1, x2])
    x12 = tf.reshape(x12, shape=(-1, 2, 128))

    g12 = GlobalAveragePooling1D(data_format='channels_first')(x12)
    # g12 = GlobalMaxPool1D(data_format='channels_first')(x12)   # Diem thap

    att12 = Dense(8, activation='relu')(g12)  # 8
    att12 = Dense(2, activation='sigmoid')(att12)
    x1_att = Lambda(lambda x: x[0] * x[1][0, 0])([x1, att12])
    x2_att = Lambda(lambda x: x[0] * x[1][0, 1])([x2, att12])

    # ====== Attention 34
    x34 = Concatenate()([x3, x4])
    x34 = tf.reshape(x34, shape=(-1, 2, 128))

    g34 = GlobalAveragePooling1D(data_format='channels_first')(x34)
    # g34 = GlobalMaxPool1D(data_format='channels_first')(x34)  # Diem thap

    att34 = Dense(8, activation='relu')(g34)  # 8
    att34 = Dense(2, activation='sigmoid')(att34)
    x3_att = Lambda(lambda x: x[0] * x[1][0, 0])([x3, att34])
    x4_att = Lambda(lambda x: x[0] * x[1][0, 1])([x4, att34])

    # ====== Combination
    mer1 = add([x1_att, x2_att])
    den1 = Dense(5, kernel_initializer=glouni,
                 activation='elu',
                 kernel_regularizer=W_regular)(mer1)  # 5 default
    den1 = BatchNormalization()(den1)
    out1 = Dropout(drop)(den1)

    mer2 = add([x3_att, x4_att])
    den2 = Dense(5, kernel_initializer=glouni,
                 activation='elu',
                 kernel_regularizer=W_regular)(mer2)  # 5 default
    den2 = BatchNormalization()(den2)
    out2 = Dropout(drop)(den2)

    # ======
    mer = add([out1, out2])  # mode='sum'
    y = Dense(4, kernel_initializer=glouni,
              activation='elu',
              kernel_regularizer=W_regular)(mer)
    y = BatchNormalization()(y)
    # y = Dropout(0.5)(y)
    out = Dense(2, kernel_initializer=glouni, activation='softmax')(y)

    final = Model(inputs=[in1, in2, in3, in4], outputs=out)

    # print(final.summary())
    tf.keras.utils.plot_model(final, "my_model.png", show_shapes=True)
    return final
