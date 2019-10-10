import numpy as np
import pandas as pd 
import sys, os, random
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from keras_bert import get_base_dict, get_model, gen_batch_inputs, load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, compile_model
import nltk, tqdm
from keras.utils import plot_model, multi_gpu_model
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Lambda, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf

config_path = 'bert_dataset/wwm_uncased_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = 'bert_dataset/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = 'bert_dataset/wwm_uncased_L-24_H-1024_A-16/vocab.txt'
def collect_inputs(abstract, tokenizer):
    done = 0
    datas = []
    for i in abstract:
        j = i.replace('$$$', ' ')
        k = tokenizer.tokenize(j)
        new_input = [k[1:-1], [ ]]
        datas.append(new_input)

        done += 1
        sys.stdout.write(str(done) + '\r')
    return datas

def generate_input_by_batch(X, batch_size = 3):
    idx = random.sample(range(len(X)), batch_size)
    X_out = [X[i] for i in idx]
    return X_out

def pretrain_model(opt_filepath, data_dir, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
    df = pd.read_csv(os.path.join(data_dir, 'task2_trainset.csv'), dtype = str)
    df_2 = pd.read_csv(os.path.join(data_dir, 'task2_public_testset.csv'), dtype = str)
    abstract_1 = df.values[:, 2]
    abstract_2 = df_2.values[:, 2]   
    token_dict = load_vocabulary(dict_path)
    token_list = list(token_dict.keys())
    tokenizer = Tokenizer(token_dict)
    X_1 = collect_inputs(abstract_1, tokenizer)
    X_2 = collect_inputs(abstract_2, tokenizer)
    X = X_1 + X_2
    print(len(X))

    start_layer, end_layer = 12, 25
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training = True, trainable = [f'Encoder-{i}-MultiHeadSelfAttention-Adapter' for i in range(start_layer, end_layer)] + [f'Encoder-{i}-FeedForward-Adapter' for i in range(start_layer, end_layer)] + [f'Encoder-{i}-MultiHeadSelfAttention-Norm' for i in range(start_layer, end_layer)] + [f'Encoder-{i}-FeedForward-Norm' for i in range(start_layer, end_layer)], seq_len = 512)
    compile_model(model)
    if os.path.exists(opt_filepath):
        model.load_weights(opt_filepath)

    def _generator():
        while True:
            yield gen_batch_inputs(generate_input_by_batch(X), token_dict, token_list, seq_len = 512, mask_rate = 0.3)
    
    checkpoint = ModelCheckpoint(opt_filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min', save_weights_only = True) 
    with open(opt_filepath+'.rc', 'r') as f:
        checkpoint.best = float(f.readline())

    es = EarlyStopping(monitor = 'val_loss', patience = 20)
    reduce_lr = ReduceLROnPlateau(factor=0.8, patience=4, verbose=1, min_lr=1e-5)
    callbacks_list = [ checkpoint, es, reduce_lr ]

    model.fit_generator(generator = _generator(), steps_per_epoch = 500, epochs = 5000, validation_data = _generator(), validation_steps = 200, callbacks = callbacks_list)


if __name__ == '__main__':
    # testing 
    opt_filepath = sys.argv[1]
    data_dir = sys.argv[2]
    pretrain_model(opt_filepath, data_dir, '1')
