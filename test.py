import numpy as np
import glob
import os
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.layers import Input, Dense, GRU, CuDNNGRU, CuDNNLSTM
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import h5py
from utils2 import Config 

# -------------------------------------------------helper function------------------------------------------------------
def language_name(index):
    if index == 0:
        return "English"
    elif index == 1:
        return "Hindi"
    elif index == 2:
        return "Mandarin"
#-----------------------------------------Inference Mode Setup----------------------------------------------------------
if __name__=='__main__':
    sequence_length = 1000
    with h5py.File("mfcc_dataset.hdf5", 'r') as hf:
        X_train = hf['X_train'][:]
        Y_train = hf['Y_train'][:]
        X_test = hf['X_test'][:]
        Y_test = hf['Y_test'][:]
    
    debug = Config['debug']
    MODEL = Config['Model']
    
    streaming_input = Input(name='streaming_input', batch_shape=(1, 1, 64))
    pred_layer1 = MODEL(64, return_sequences=True, name='layer1', stateful=True)(streaming_input)
    pred_layer2 = MODEL(32, return_sequences=True, name='layer2')(pred_layer1)
    pred_layer3 = Dense(100, activation='tanh', name='layer3')(pred_layer2)
    pred_output = Dense(3, activation='softmax', name='rnn_output')(pred_layer3)
    streaming_model = Model(inputs=streaming_input, outputs=pred_output)
    streaming_model.load_weights('model_GRU.hdf5')
    streaming_model.summary()
    
    if debug:
        # Language Prediction for a random sequence from the validation data set
        random_sample = np.random.randint(0, X_test.shape[0])
        random_sequence_num = np.random.randint(0, len(X_test[random_sample]))
        test_single = X_test[random_sample][random_sequence_num].reshape(1, 1, 64)
        test_label = Y_test[random_sample][random_sequence_num]
        true_label = language_name(np.argmax(test_label))
        print("*"*50)
        print("True label is ", true_label)
        pred = streaming_model.predict(test_single)
        pred_label = language_name(np.argmax(pred))
        print("Predicted label is ", pred_label)
        print("*"*50)
    else:
        # Prediction for all sequences in the validation set - Takes very long to run
        print("Predicting labels for all sequences - (Will take a lot of time)")
        list_pred_labels = []
        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                test = X_test[i][j].reshape(1, 1, 64)
                seq_predictions_prob = streaming_model.predict(test)
                predicted_language_index = np.argmax(seq_predictions_prob)
                list_pred_labels.append(predicted_language_index)
        pred_english = list_pred_labels.count(0)
        pred_hindi = list_pred_labels.count(1)
        pred_mandarin = list_pred_labels.count(2)
        print("Number of English labels = ", pred_english)
        print("Number of Hindi labels = ", pred_hindi)
        print("Number of Mandarin labels = ", pred_mandarin)