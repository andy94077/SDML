import numpy as np
import pandas as pd
from keras.models import load_model
import sys, os 
from tqdm import tqdm
from keras_bert import get_base_dict, get_model, gen_batch_inputs, load_vocabulary, load_trained_model_from_checkpoint, Tokenizer
from keras.layers import Lambda, Dense
from keras.models import Model
from keras import backend as K
import tensorflow as tf


def get_model(model_path, config_path, checkpoint_path):
    # getting model:
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training = True, trainable = True, seq_len = 512)
    Input_layer = model.inputs[:2]
    x = model.layers[-9].output
    x = Lambda(lambda model: model[:, 0])(x)
    Output_layer = Dense(3, activation = 'sigmoid')(x)
    model = Model(Input_layer, Output_layer)
    model.load_weights(model_path)
    return model

def load_task2_testX(dict_path):
    if not os.path.exists('task2_testX.npy') or not os.path.exists('task2_test_seg.npy'):
        df = pd.read_csv('task2_public_testset.csv', dtype = str)
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

        np.save('task2_testX.npy', X)
        np.save('task2_test_seg.npy', seg)
    else:
        X, seg = np.load('task2_testX.npy'), np.load('task2_test_seg.npy')
    return X, seg

def generate_submit(model, output_file, dict_path):
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
