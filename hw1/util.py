import numpy as np
import pandas as pd 
import sys, os
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from keras_bert import get_base_dict, get_model, gen_batch_inputs, load_vocabulary, load_trained_model_from_checkpoint, Tokenizer
from tqdm import tqdm

def load_task2_trainXY(dict_path, data_dir):
    if not os.path.exists(os.path.join(data_dir, 'task2_trainX.npy')) or not os.path.exists(os.path.join(data_dir, 'task2_trainY.npy')) or not os.path.exists(os.path.join(data_dir, 'task2_train_seg.npy')):
        df = pd.read_csv(os.path.join(data_dir, 'task2_trainset.csv'), dtype = str)
        cate = df.values[:, -1] 

        # generating Y
        Y = np.zeros((cate.shape[0], 4))
        name = {'THEORETICAL':0, 'ENGINEERING':1, 'EMPIRICAL':2, 'OTHERS':3}
        for i in range(cate.shape[0]):
            for c in cate[i].split(' '):
                Y[i, name[c]] += 1

        # generating X
        abstract = df.values[:, 2]

        # collect words
        token_dict = load_vocabulary(dict_path)
        tokenizer = Tokenizer(token_dict)
        input_data = []
        input_seg = []
        for i in tqdm(abstract):
            j = i.replace('$$$', ' ')
            idx, seg = tokenizer.encode(j, max_len=512)
            input_data.append(idx)
            input_seg.append(seg)
        X = np.array(input_data)
        seg = np.array(input_seg)
        np.save(os.path.join(data_dir, 'task2_trainX.npy'), X)
        np.save(os.path.join(data_dir, 'task2_trainY.npy'), Y)
        np.save(os.path.join(data_dir, 'task2_train_seg.npy'), seg)
    else:
        X, Y, seg = np.load(os.path.join(data_dir, 'task2_trainX.npy')), np.load(os.path.join(data_dir, 'task2_trainY.npy')), np.load(os.path.join(data_dir, 'task2_train_seg.npy'))
    return X, Y, seg
