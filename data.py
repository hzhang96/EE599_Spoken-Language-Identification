import numpy as np
from utils2 import Config 
import os
import h5py
from sklearn.model_selection import train_test_split



# ----------------------------------------------helper function---------------------------------------------------
def get_features(path):
    import librosa as lib
    y, sr = lib.load(path, sr = 16000)
    mat = lib.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
    #print(y.shape, sr, mat.shape)
    return mat

if __name__=='__main__':
    path = Config['input_path']
    num_mfcc_features = Config['num_mfcc_features']
    size = Config['testset_size']

    eng_path = os.path.join(path, "train_english")
    hin_path = os.path.join(path, "train_hindi")
    man_path = os.path.join(path, "train_mandarin")

    english_mfcc = np.array([]).reshape(0, num_mfcc_features)
    hindi_mfcc = np.array([]).reshape(0, num_mfcc_features)
    mandarin_mfcc = np.array([]).reshape(0, num_mfcc_features)

    for dirname, _, filenames in os.walk(eng_path):
        for filename in filenames:
            cur_data = get_features(os.path.join(dirname, filename)).T
            english_mfcc = np.vstack((english_mfcc, cur_data))
        
    for dirname, _, filenames in os.walk(hin_path):
        for filename in filenames:
            cur_data = get_features(os.path.join(dirname, filename)).T
            hindi_mfcc = np.vstack((hindi_mfcc, cur_data))
                
    for dirname, _, filenames in os.walk(man_path):
        for filename in filenames:
            cur_data = get_features(os.path.join(dirname, filename)).T
            mandarin_mfcc = np.vstack((mandarin_mfcc, cur_data))
        
    # Sequence length is 10 seconds 
    sequence_length = 1000
    list_english_mfcc = []
    num_english_sequence = int(np.floor(len(english_mfcc)/sequence_length))
    for i in range(num_english_sequence):
        list_english_mfcc.append(english_mfcc[sequence_length*i:sequence_length*(i+1)])
    list_english_mfcc = np.array(list_english_mfcc)
    english_labels = np.full((num_english_sequence, 1000, 3), np.array([1, 0, 0]))

    list_hindi_mfcc = []
    num_hindi_sequence = int(np.floor(len(hindi_mfcc)/sequence_length))
    for i in range(num_hindi_sequence):
        list_hindi_mfcc.append(hindi_mfcc[sequence_length*i:sequence_length*(i+1)])
    list_hindi_mfcc = np.array(list_hindi_mfcc)
    hindi_labels = np.full((num_hindi_sequence, 1000, 3), np.array([0, 1, 0]))

    list_mandarin_mfcc = []
    num_mandarin_sequence = int(np.floor(len(mandarin_mfcc)/sequence_length))
    for i in range(num_mandarin_sequence):
        list_mandarin_mfcc.append(mandarin_mfcc[sequence_length*i:sequence_length*(i+1)])
    list_mandarin_mfcc = np.array(list_mandarin_mfcc)
    mandarin_labels = np.full((num_mandarin_sequence, 1000, 3), np.array([0, 0, 1]))

    del english_mfcc
    del hindi_mfcc
    del mandarin_mfcc        
        
    total_sequence_length = num_english_sequence + num_hindi_sequence + num_mandarin_sequence
    Y_train = np.vstack((english_labels, hindi_labels))
    Y_train = np.vstack((Y_train, mandarin_labels))

    X_train = np.vstack((list_english_mfcc, list_hindi_mfcc))
    X_train = np.vstack((X_train, list_mandarin_mfcc))

    del list_english_mfcc
    del list_hindi_mfcc
    del list_mandarin_mfcc

    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=size)

    with h5py.File("mfcc_dataset.hdf5", 'w') as hf:
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('Y_train', data=Y_train)
    hf.create_dataset('X_test', data=X_val)
    hf.create_dataset('Y_test', data=Y_val)    
        
