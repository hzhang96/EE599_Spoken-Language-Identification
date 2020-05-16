import numpy as np
import glob
import os
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.layers import CuDNNGRU, CuDNNLSTM
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import h5py
from utils2 import Config 

#--------------------------------------------------load dataset----------------------------------------------------------------
if __name__=='__main__':
    with h5py.File("mfcc_dataset.hdf5", 'r') as hf:
        X_train = hf['X_train'][:]
        Y_train = hf['Y_train'][:]
        X_test = hf['X_test'][:]
        Y_test = hf['Y_test'][:]

#----------------------------------------Setting up the model for training-----------------------------------------------------
    sequence_length = 1000
    DROPOUT = 0.3
    RECURRENT_DROP_OUT = 0.2
    optimizer = optimizers.Adam(decay=1e-4)
    main_input = Input(shape=(sequence_length, 64), name='main_input')

    # ### main_input = Input(shape=(None, 64), name='main_input')
    # ### pred_gru = GRU(4, return_sequences=True, name='pred_gru')(main_input)
    # ### rnn_output = Dense(3, activation='softmax', name='rnn_output')(pred_gru)

    MODEL = Config['Model']
    layer1 = MODEL(64, return_sequences=True, name='layer1')(main_input)
    layer2 = MODEL(32, return_sequences=True, name='layer2')(layer1)
    layer3 = Dense(100, activation='tanh', name='layer3')(layer2)
    rnn_output = Dense(3, activation='softmax', name='rnn_output')(layer3)

    model = Model(inputs=main_input, outputs=rnn_output)
    print('\nModel Compiling...')
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    model.summary()
    history = model.fit(X_train, Y_train, batch_size=32, epochs=75, validation_data=(X_test, Y_test), shuffle=True, verbose=1)
    model.save('model_LSTM.hdf5')
    print('\nModel already saved as model_GRU.hdf5')