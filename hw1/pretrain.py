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

def get_layers_name(attention_layers):
    if isinstance(attention_layers, int):
        attention_layers = [attention_layers]
    return [f'Encoder-{i}-MultiHeadSelfAttention-Adapter' for i in attention_layers] + [f'Encoder-{i}-FeedForward-Adapter' for i in attention_layers] + [f'Encoder-{i}-MultiHeadSelfAttention-Norm' for i in attention_layers] + [f'Encoder-{i}-FeedForward-Norm' for i in attention_layers]

def pretrain_model(opt_filepath, data_dir, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id 
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    
    token_dict = load_vocabulary(dict_path)
    token_list = list(token_dict.keys())
    #if not os.path.exists(os.path.join(data_dir, 'pretrain_X.npy')):
    df = pd.read_csv(os.path.join(data_dir, 'task2_trainset.csv'), dtype = str)
    df_2 = pd.read_csv(os.path.join(data_dir, 'task2_public_testset.csv'), dtype = str)
    abstract_1 = df.values[:, 2]
    abstract_2 = df_2.values[:, 2]   
    tokenizer = Tokenizer(token_dict)
    X_1 = collect_inputs(abstract_1, tokenizer)
    X_2 = collect_inputs(abstract_2, tokenizer)
    X = np.array(X_1 + X_2)
    #    np.save(os.path.join(data_dir, 'pretrain_X.npy'), X)
    #else:
    #    X = np.load(os.path.join(data_dir, 'pretrain_X.npy'))
    print(X.shape)

    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, trainable=get_layers_name(range(12,25)), seq_len=512)
    compile_model(model)

    def _generator(batch_size=4):
        while True:
            idx = np.random.permutation(X.shape[0])
            for i in range(0, idx.shape[0], batch_size):
                yield gen_batch_inputs(X[i:i+batch_size], token_dict, token_list, seq_len = 512, mask_rate = 0.3)
    
    checkpoint = ModelCheckpoint(opt_filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min', save_weights_only = True) 

    trainable_layer = list(range(12*8, 19*8, 8))
    batch_size = [3]*3 + [3]*3
    for i, layer_i in enumerate(trainable_layer):
        for j, layer in enumerate(model.layers):
            if j >= layer_i:
                layer.trainable = True
                print(layer.name, layer.trainable)
            else: 
                layer.trainable = False

        compile_model(model)
        if os.path.exists(opt_filepath):
            model.load_weights(opt_filepath)
        
        es = EarlyStopping(monitor = 'val_loss', patience = 20)
        reduce_lr = ReduceLROnPlateau(factor=0.7, patience=4, verbose=1, min_lr=1e-6)
        callbacks_list = [ checkpoint, es, reduce_lr ]

        model.fit_generator(generator = _generator(batch_size[i]), steps_per_epoch = 500, epochs = 5000, validation_data = _generator(), validation_steps = 200, callbacks = callbacks_list)


if __name__ == '__main__':
    # testing 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    opt_filepath = sys.argv[1]
    data_dir = sys.argv[2]
    pretrain_model(opt_filepath, data_dir, '2')
