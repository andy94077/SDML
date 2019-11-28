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

#def decode_sequence(encoder_model, decoder_model, testX, testX2, test_ith, test_ith2, test_ith_str, test_ith_str2, test_word, test_word2, max_seq_len, word2idx):
def decode_sequence(encoder_model, decoder_model, testXs, test_iths, test_ith_strs, test_words, setting, setting2, max_seq_len, word2idx):
    encoder_out, state = encoder_model.predict_on_batch([testXs[0], test_ith_strs[0], test_words[0]])
    encoder_out2, state2 = encoder_model.predict_on_batch([testXs[1], test_ith_strs[1], test_words[1]])
    encoder_outs = [encoder_out, encoder_out2]
    states = [state, state2]
    state, state2 = state.copy(), state2.copy()

    target_seq = np.full((testXs[0].shape[0], max_seq_len+1), word2idx[''], np.int32)
    target_seq[:, 0] = word2idx['<SOS>']
    seq_eos = np.zeros(testXs[0].shape[0], np.bool)
    for i in range(max_seq_len):
        decoder_out, states[setting[0]] = decoder_model.predict_on_batch([target_seq[:, i:i+1], encoder_outs[setting[1]], states[setting[0]], test_iths[setting[2]], test_words[setting[3]]])
        target_seq[:, i+1] = np.argmax(decoder_out[:, 0], axis=-1)
        seq_eos[target_seq[:, i+1] == word2idx['<EOS>']] = True
        if np.all(seq_eos):
            break
    else:
        target_seq[np.logical_not(seq_eos), -1] = word2idx['<EOS>']

    states = [state, state2]
    target_seq2 = np.full((testXs[1].shape[0], max_seq_len+1), word2idx[''], np.int32)
    target_seq2[:, 0] = word2idx['<SOS>']
    seq_eos = np.zeros(testXs[1].shape[0], np.bool)
    for i in range(max_seq_len):
        decoder_out, states[setting2[0]] = decoder_model.predict_on_batch([target_seq2[:, i:i+1], encoder_outs[setting2[1]], states[setting2[0]], test_iths[setting2[2]], test_words[setting2[3]]])
        target_seq2[:, i+1] = np.argmax(decoder_out[:, 0], axis=-1)
        seq_eos[target_seq2[:, i+1] == word2idx['<EOS>']] = True
        if np.all(seq_eos):
            break
    else:
        target_seq2[np.logical_not(seq_eos), -1] = word2idx['<EOS>']
    return target_seq, target_seq2

if __name__ == '__main__':
    max_seq_len = 32 + 2 #contain <SOS> and <EOS>

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('model_path')
    parser.add_argument('submit_i1w1')
    parser.add_argument('submit_i2w2')
    parser.add_argument('setting')
    parser.add_argument('setting2')
    args = parser.parse_args()
    data = args.data
    model_path = args.model_path
    submit_i1w1 = args.submit_i1w1
    submit_i2w2 = args.submit_i2w2
    setting = list(map(int, args.setting))
    setting2 = list(map(int, args.setting2))

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
    with open(submit_i1w1, 'r') as t:
        input_testX1 = [line.split() for line in t.readlines()]
        testX = [[word2idx[w] if w in word2idx else word2idx[''] for w in ll[:-2]] for ll in input_testX1]
        testX = pad_sequences(testX, max_seq_len, padding='post', truncating='post', value=word2idx[''])
        
        #the index of the word in testX starts from 1, which is different from our model
        test_ith = np.array([[int(ll[-2])-1] for ll in input_testX1], dtype=np.int32)
        test_ith_str = np.array([[word2idx[str(i)]] for i in test_ith.ravel()], dtype=np.int32)
        test_word = np.array([[word2idx[ll[-1]] if ll[-1] in word2idx else word2idx['']] for ll in input_testX1], dtype=np.int32)

    with open(submit_i2w2, 'r') as t2:
        input_testX2 = [line.split() for line in t2.readlines()]
        testX2 = [[word2idx[w] if w in word2idx else word2idx[''] for w in ll[:-2]] for ll in input_testX2]
        testX2 = pad_sequences(testX2, max_seq_len, padding='post', truncating='post', value=word2idx[''])
        
        #the index of the word in testX2 starts from 1, which is different from our model
        test_ith2 = np.array([[int(ll[-2])-1] for ll in input_testX2], dtype=np.int32)
        test_ith_str2 = np.array([[word2idx[str(i)]] for i in test_ith2.ravel()], dtype=np.int32)
        test_word2 = np.array([[word2idx[ll[-1]] if ll[-1] in word2idx else word2idx['']] for ll in input_testX2], dtype=np.int32)

    testXs, test_iths, test_ith_strs, test_words = np.array([testX, testX2]), np.array([test_ith, test_ith2]), np.array([test_ith_str, test_ith_str2]), np.array([test_word, test_word2]) 
    with open('output.txt.i1w2', 'w') as f, open('output.txt.i2w1', 'w') as f2:
        batch_size = 1024
        for i in trange(0, testX.shape[0], batch_size):
            decoder_seq, decoder_seq2 = decode_sequence(encoder_model, decoder_model, testXs[:, i:i+batch_size], test_iths[:, i:i+batch_size], test_ith_strs[:, i:i+batch_size], test_words[:, i:i+batch_size], setting, setting2, max_seq_len, word2idx)
            print(*[' '.join([idx2word[idx] for idx in ll]).strip() for ll in decoder_seq], sep='\n', file=f)
            print(*[' '.join([idx2word[idx] for idx in ll]).strip() for ll in decoder_seq2], sep='\n', file=f2)

    os.system(f'python3 data/hw2.1_evaluate.py --training_file {submit_i1w1} --result_file output.txt.i1w2')
    #os.system(f'python3 data/hw2.1_evaluate.py --training_file {submit_i2w2} --result_file output.txt.i2w1')

