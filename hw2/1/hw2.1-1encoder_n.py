import os, sys, argparse
import numpy as np
from tqdm import trange
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, RepeatVector, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.version.VERSION == '2.0.0':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)


def build_model(hidden_dim, max_seq_len, vocabulary_size):
    ## encoder Input and layers
    encoder_in = Input((max_seq_len,), dtype='int32', name='encoder_in')
    ith_str = Input((1,), dtype='int32', name='ith_str')
    word = Input((1,), dtype='int32', name='word')
    OneHot = Lambda(lambda x: K.one_hot(x, vocabulary_size), name='OneHot')

    ## building encoder
    encoder_in_and_word = Concatenate()([ith_str, word, encoder_in])
    encoder_GRU = GRU(hidden_dim, return_state=True)
    encoder_out, state = encoder_GRU(OneHot(encoder_in_and_word))
    encoder_out_dup = RepeatVector(max_seq_len)(encoder_out)

    ## decoder Input and layers
    decoder_in = Input((max_seq_len,), dtype='int32', name='decoder_in')
    ith = Input((1,), dtype='int32', name='ith')
    decoder_GRU = GRU(hidden_dim, return_sequences=True, return_state=True)
    decoder_Dense = Dense(vocabulary_size, activation='softmax', name='decoder_out')

    ## building decoder
    ith_dup = RepeatVector(max_seq_len)(K.cast(ith, 'float'))
    word_dup = K.reshape(RepeatVector(max_seq_len)(word), (-1, max_seq_len))
    x = Concatenate()([ith_dup, OneHot(word_dup), OneHot(decoder_in), encoder_out_dup])
    x, _ = decoder_GRU(x, initial_state=state)
    decoder_out = decoder_Dense(x)

    ## get the specific word
    gather = K.concatenate([K.reshape(tf.range(K.shape(decoder_out)[0]), (-1, 1)), ith])
    specific_word = tf.gather_nd(decoder_out, gather)
    specific_word = Lambda(tf.identity, name='word_out')(specific_word) # Add this layer because the name of tf.gather_nd is too ugly

    model = Model([encoder_in, decoder_in, ith, ith_str, word], [decoder_out, specific_word])

    ## building decoder model given encoder_out and states
    decoder_in_one_word = Input((1,), dtype='int32', name='decoder_in_one_word')
    decoder_state_in = Input((hidden_dim,), name='decoder_state_in')
    encoder_out = Input((hidden_dim,), name='decoder_encoder_out')
    x = Concatenate()([K.cast(ith, 'float')[:, tf.newaxis], OneHot(word), OneHot(decoder_in_one_word), encoder_out[:, tf.newaxis]])
    x, decoder_state = decoder_GRU(x, initial_state=decoder_state_in)
    decoder_out = decoder_Dense(x)
    decoder_model = Model([decoder_in_one_word, encoder_out, decoder_state_in, ith, word], [decoder_out, decoder_state])

    encoder_in = Input((None,), dtype='int32')
    encoder_in_and_word = Concatenate()([ith_str, word, encoder_in])
    encoder_out, state = encoder_GRU(OneHot(encoder_in_and_word))
    encoder_model = Model([encoder_in, ith_str, word], [encoder_out, state])
    return model, encoder_model, decoder_model

def generate_word(Y, line_len):
    word_idx = np.array([np.random.randint(0, line_len[i]-1) for i in range(Y.shape[0])]) # do not include <SOS>
    return [word_idx.reshape(-1, 1), np.array([[word2idx[str(i)]] for i in word_idx]), Y[np.arange(Y.shape[0]), word_idx].reshape(-1, 1)]

def decode_sequence(encoder_model, decoder_model, testX, test_ith, test_ith_str, test_word, max_seq_len, word2idx):
    encoder_out, state = encoder_model.predict_on_batch([testX, test_ith_str, test_word])
    target_seq = np.full((testX.shape[0], max_seq_len+1), word2idx[''], np.int32)
    target_seq[:, 0] = word2idx['<SOS>']
    seq_eos = np.zeros(testX.shape[0], np.bool)
    for i in range(max_seq_len):
        decoder_out, state = decoder_model.predict_on_batch([target_seq[:, i:i+1], encoder_out, state, test_ith, test_word])
        target_seq[:, i+1] = np.argmax(decoder_out[:, 0], axis=-1)
        seq_eos[target_seq[:, i+1] == word2idx['<EOS>']] = True
        if np.all(seq_eos):
            break
    else:
        target_seq[np.logical_not(seq_eos), -1] = word2idx['<EOS>']
    return target_seq

if __name__ == '__main__':
    max_seq_len = 32 + 2 #contain <SOS> and <EOS>

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('model_path')
    parser.add_argument('submit')
    parser.add_argument('-n', '--encoder-num', type=int, default=max_seq_len)
    args = parser.parse_args()
    data = args.data
    model_path = args.model_path
    submit = args.submit
    encoder_num = args.encoder_num

    with open(data, 'r') as f:
        inputX = [line.strip() for line in f.readlines()]#[:200000]
    word2idx = { w: i for i, w in enumerate(np.unique([item for ll in inputX for item in ll] + ['', '<SOS>', '<EOS>'] + [str(i) for i in range(max_seq_len)]))}
    idx2word = {word2idx[w]: w for w in word2idx}
    vocabulary_size = len(word2idx)
    print(f'\033[32;1mvocabulary_size: {vocabulary_size}\033[0m')

    hidden_dim = 128
    model, encoder_model, decoder_model= build_model(hidden_dim, max_seq_len, vocabulary_size)

    print('\033[32;1mLoading Model\033[0m')
    model.load_weights(model_path)
    if submit:
        with open('output.txt', 'w') as f, open(submit, 'r') as t:
            input_testX = [line.split() for line in t.readlines()]
            testX = [[word2idx[w] if w in word2idx else word2idx[''] for w in ll[:-2]] for ll in input_testX]
            testX = pad_sequences(testX, max_seq_len, padding='post', truncating='post', value=word2idx[''])
            
            #the index of the word in testX starts from 1, which is different from our model
            test_ith = np.array([[int(ll[-2])-1] for ll in input_testX], dtype=np.int32)
            test_ith_str = np.array([[word2idx[str(i)]] for i in test_ith.ravel()], dtype=np.int32)
            test_word = np.array([[word2idx[ll[-1]] if ll[-1] in word2idx else word2idx['']] for ll in input_testX], dtype=np.int32)

            batch_size = 1024
            for i in trange(0, testX.shape[0], batch_size):
                decoder_seq = decode_sequence(encoder_model, decoder_model, testX[i:i+batch_size, :encoder_num], test_ith[i:i+batch_size], test_ith_str[i:i+batch_size], test_word[i:i+batch_size], max_seq_len, word2idx)
                print(*[' '.join([idx2word[idx] for idx in ll]).strip() for ll in decoder_seq], sep='\n', file=f)
            os.system(f'python3 data/hw2.1_evaluate.py --training_file {submit} --result_file output.txt')

