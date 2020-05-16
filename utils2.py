import numpy as np
import os
import os.path as osp
import argparse
from tensorflow.compat.v1.keras.layers import Input, Dense, GRU, CuDNNGRU, CuDNNLSTM

Config ={}
Config['input_path'] = r'D:\StudyMaterial\USC\Semester3_20spring\EE_599_Deep learning\HW5\train'
Config['num_mfcc_features'] = 64
Config['use_cuda'] = True
Config['testset_size'] = 0.2
Config['debug'] = True
Config['Model'] = CuDNNGRU