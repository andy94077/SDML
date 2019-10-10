import numpy as np
import pandas as pd 
import sys, os
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from keras_bert import get_base_dict, get_model, gen_batch_inputs, load_vocabulary, load_trained_model_from_checkpoint, Tokenizer
from tqdm import tqdm
from keras.utils import plot_model, multi_gpu_model
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Lambda, Dense, BatchNormalization, Dropout, Activation, Concatenate, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow.compat.v1 as tf

import util

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

bert_version = 12
if bert_version == 24:
    config_path = 'bert_dataset/wwm_uncased_L-24_H-1024_A-16/bert_config.json'
    checkpoint_path = 'bert_dataset/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt'
    dict_path = 'bert_dataset/wwm_uncased_L-24_H-1024_A-16/vocab.txt'
else:
    config_path = 'bert_dataset/uncased_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'bert_dataset/uncased_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'bert_dataset/uncased_L-12_H-768_A-12/vocab.txt'
    
seq_len = 512
opt_filepath = sys.argv[1]
data_dir = sys.argv[2]

X, Y, seg = util.load_task2_trainXY(dict_path, data_dir)
citation_count = np.load(os.path.join(data_dir, 'citation_count_train.npy'))
assert X.shape[0] == Y.shape[0]
np.random.seed(246601)
rd_seq = np.random.permutation(X.shape[0])
X_train, X_val, Y_train, Y_val = X[rd_seq[:-1000]], X[rd_seq[-1000:]], Y[rd_seq[:-1000]], Y[rd_seq[-1000:]]
seg_train, seg_val = seg[rd_seq[:-1000]], seg[rd_seq[-1000:]]
citation_count_train, citation_count_val = citation_count[rd_seq[:-1000]], citation_count[rd_seq[-1000:]]
print(f'\033[32;1mX_train: {X_train.shape}, X_val:{X_val.shape}, Y_train:{Y_train.shape}, Y_val:{Y_val.shape}, seg_train:{seg_train.shape}, seg_val:{seg_val.shape}\033[0m')

def f1_acc(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis = 0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis = 0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis = 0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis = 0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis = 0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis = 0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training = True, trainable = True, seq_len = seq_len)
#model.load_weights('fine_tune/model12-24single.weight')
model.load_weights('bert_dataset/bert_custom_pretrained_v4.weight')

Input_layer = model.inputs[:2]
x = model.layers[-9].output
x = Lambda(lambda model: model[:, 0])(x)
c = Input(shape=(4,))
x = Concatenate()([x, c])
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('tanh')(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('tanh')(x)
Output_layer = Dense(3, activation = 'sigmoid')(x)
model = Model(Input_layer+[c], Output_layer)

checkpoint = ModelCheckpoint(opt_filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min', save_weights_only = True) 
reduce_lr = ReduceLROnPlateau(factor=0.7, patience=4, verbose=1, min_lr=1e-6)
callbacks_list = [checkpoint, reduce_lr]

#model.summary()

trainable_layer = [199] if bert_version == 24 else [103]#, 87, 71, 55]
epoch_num = [80, 40, 40, 8 , 4]
batch_size = [8, 8, 8, 8, 4]
resume = False
st = -1
if resume:
    if os.path.exists(opt_filepath+'.rc'):
        print('\033[32;1mLoad Model\033[0m')
        with open(opt_filepath+'.rc', 'r') as f:
            st = int(f.readline())
            checkpoint.best = float(f.readline())
    for l in range(len(trainable_layer))[st+1:]:
        for i, layer in enumerate(model.layers):
            if i > trainable_layer[l]:
                layer.trainable = True
                print(layer.name, layer.trainable)
            else: 
                layer.trainable = False

        model.compile(loss=f1_loss, optimizer = Adam(1e-3), metrics = [f1_acc, 'acc'])
        if os.path.exists(opt_filepath):
            model.load_weights(opt_filepath)

        model.fit([X_train, seg_train, citation_count_train], Y_train[:, :-1], batch_size=batch_size[l], epochs=epoch_num[l], callbacks=callbacks_list, validation_data=([X_val, seg_val, citation_count_val], Y_val[:, :-1]))

        with open(opt_filepath+'.rc', 'w') as f:
            f.write(f'{l}\n{checkpoint.best}\n')
        model.save_weights(opt_filepath+f'_{trainable_layer[l]}')
else:
    output_file = sys.argv[3]
    print(f'\033[32;1mGenerating output file {output_file}...\033[0m')
    model.load_weights(opt_filepath)
    #print(model.evaluate([X_val, seg_val, citation_count_val], Y_val[:, :-1]))

    X_test, seg_test = util.load_task2_testX(dict_path, data_dir)
    c_test = np.load(os.path.join(data_dir, 'citation_count_test.npy'))

    Y_pred = model.predict([X, seg, citation_count], verbose=1)
    np.save(output_file+'_pred_train.npy', Y_pred)

    Y_pred = model.predict([X_test, seg_test, c_test], verbose=1)
    np.save(output_file+'_pred_test.npy', Y_pred)
    #util.generate_submit(model, output_file, dict_path, data_dir)

