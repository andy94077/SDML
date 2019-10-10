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

def load_task2_testX(dict_path, data_dir):
    if not os.path.exists(os.path.join(data_dir, 'task2_testX.npy')) or not os.path.exists(os.path.join(data_dir, 'task2_test_seg.npy')):
        df = pd.read_csv(os.path.join(data_dir, 'task2_public_testset.csv'), dtype = str)
        abstract = df.values[:, 2]

        # collect words
        token_dict = load_vocabulary(dict_path)
        tokenizer = Tokenizer(token_dict)
        input_data = []
        input_seg = []
        seq_len = 512 # maximum should be 638, while bert-BASE only support up to 512
        for i in tqdm(abstract):
            j = i.replace('$$$', ' ')
            idx, seg = tokenizer.encode(j, max_len=seq_len)
            input_data.append(idx)
            input_seg.append(seg)
        X = np.asarray(input_data)
        seg = np.asarray(input_seg)

        np.save(os.path.join(data_dir, 'task2_testX.npy'), X)
        np.save(os.path.join(data_dir, 'task2_test_seg.npy'), seg)
    else:
        X, seg = np.load(os.path.join(data_dir, 'task2_testX.npy')), np.load(os.path.join(data_dir, 'task2_test_seg.npy'))
    return X, seg

def generate_submit(model, output_file, dict_path, data_dir):
    X, seg = load_task2_testX(dict_path)
    Y_pred = model.predict([X, seg], verbose=1)
    Y_pred = (Y_pred > 0.5)
    other_pred = np.sum(Y_pred, axis=1) < 0.9
    Y = np.hstack((Y_pred, other_pred.reshape(-1, 1))).astype('int')

    with open(output_file, 'w') as f:
        f.write('order_id,THEORETICAL,ENGINEERING,EMPIRICAL,OTHERS\n')
        for i in range(Y_pred.shape[0]):
            print('T' + str(i+1).zfill(5), Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3], sep=',', file=f)

        for i in range(Y_pred.shape[0]):
            f.write('T' + str(i + 20001).zfill(5) + ',0,0,0,0\n')
