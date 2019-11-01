import os, sys, argparse
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, RepeatVector, Lambda, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
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


def train_data_preprocessing(inputX, word2idx, max_seq_len):
    X = [[word2idx['<SOS>']]+[word2idx[w] for w in ll[:min(len(ll), max_seq_len-2)]]+[word2idx['<EOS>']] for ll in inputX]
    trainX, Y = X[:-1], X[1:]
    line_len = np.array([min(len(i), max_seq_len) for i in Y])
    trainX = pad_sequences(trainX, max_seq_len, padding='post', truncating='post', value=word2idx[''])
    Y = pad_sequences(Y, max_seq_len+1, padding='post', truncating='post', value=word2idx[''])
    np.random.seed(880301)
    idx = np.random.permutation(trainX.shape[0])
    train_seq, valid_seq = idx[:int(trainX.shape[0]*0.9)], idx[int(trainX.shape[0]*0.9):]
    trainX, validX = trainX[train_seq], trainX[valid_seq]
    trainY_SOS, validY_SOS = Y[train_seq, :-1], Y[valid_seq, :-1]
    trainY, validY = Y[train_seq, 1:], Y[valid_seq, 1:]
    line_len, valid_line_len = line_len[train_seq], line_len[valid_seq]
    return trainX, validX, trainY_SOS, validY_SOS, trainY, validY, line_len, valid_line_len


def build_model(hidden_dim, max_seq_len, vocabulary_size):
    ## encoder Input and layers
    encoder_in = Input((max_seq_len,), dtype='int32', name='encoder_in')
    ith_str = Input((2,), dtype='int32', name='ith_str')
    word = Input((2,), dtype='int32', name='word')
    OneHot = Lambda(lambda x: K.one_hot(x, vocabulary_size), name='OneHot')

    ## building encoder
    encoder_in_and_word = Concatenate()([ith_str, word, encoder_in])
    encoder_out, state = GRU(hidden_dim, return_state=True)(OneHot(encoder_in_and_word))
    encoder_out = RepeatVector(max_seq_len)(encoder_out)

    ## decoder Input and layers
    decoder_in = Input((max_seq_len,), dtype='int32', name='decoder_in')
    ith = Input((2,), dtype='int32', name='ith')
    decoder_GRU = GRU(hidden_dim, return_sequences=True)
    decoder_Dense = Dense(vocabulary_size, activation='softmax', name='decoder_out')

    ## building decoder
    ith_dup = RepeatVector(max_seq_len)(K.cast(ith, 'float'))
    word_one_hot = K.reshape(OneHot(word), (-1, 2*vocabulary_size))
    word_dup = RepeatVector(max_seq_len)(word_one_hot)
    x = Concatenate()([ith_dup, word_dup, OneHot(decoder_in), encoder_out])
    x = decoder_GRU(x, initial_state=state)
    decoder_out = decoder_Dense(x)

    ## get the specific word
    gather = K.concatenate([K.reshape(tf.math.floordiv(tf.range(2*K.shape(decoder_out)[0]), 2), (-1, 1)), K.reshape(ith, (-1, 1))])
    specific_word = tf.gather_nd(decoder_out, gather)
    specific_word = Lambda(lambda x: K.reshape(x, (-1, 2, vocabulary_size)), name='word_out')(specific_word)

    model = Model([encoder_in, decoder_in, ith, ith_str, word], [decoder_out, specific_word])
    encoder_model = Model([encoder_in, ith_str, word], [encoder_out, state])

    ## building decoder model given encoder_out and states
    decoder_state_in = Input((hidden_dim,), name='decoder_state_in')
    encoder_out = Input((max_seq_len, hidden_dim), name='decoder_encoder_out')
    x = Concatenate()([ith_dup, word_dup, OneHot(decoder_in), encoder_out])
    x = decoder_GRU(x, initial_state=decoder_state_in)
    decoder_out = decoder_Dense(x)
    decoder_model = Model([decoder_in, encoder_out, decoder_state_in, ith, word], decoder_out)
    return model, encoder_model, decoder_model

def generate_word(Y, line_len):
    word_idx = np.array([np.random.randint(0, line_len[i]-1, size=2) for i in range(Y.shape[0])]) # do not include <SOS>
    return [word_idx, np.array([[word2idx[str(i[0])], word2idx[str(i[1])]] for i in word_idx]), Y[np.arange(2*Y.shape[0])//2, word_idx.ravel()].reshape(-1, 2)]

def decode_sequence(encoder_model, decoder_model, testX, test_ith, test_ith_str, test_word, max_seq_len, word2idx):
    encoder_out, state = encoder_model.predict([testX, test_ith_str, test_word], batch_size=256)
    target_seq = np.full((1, max_seq_len+1), word2idx[''], np.int32)
    target_seq[0, 0] = word2idx['<SOS>']
    for i in range(max_seq_len):
        decoder_out = decoder_model.predict_on_batch([target_seq[:, :-1], encoder_out, state, test_ith, test_word])
        target_seq[0, i+1] = np.argmax(decoder_out[0, i])
        if target_seq[0, i+1] == word2idx['<EOS>']:
            break
    else:
        target_seq[0, -1] = word2idx['<EOS>']
    return target_seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('model_path')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--submit')
    args = parser.parse_args()
    data = args.data
    model_path = args.model_path
    training = not args.no_training
    submit = args.submit

    max_seq_len = 32 + 2 #contain <SOS> and <EOS>
    with open(data, 'r') as f:
        inputX = [line.strip() for line in f.readlines()]#[:200000]
    word2idx = { w: i for i, w in enumerate(np.unique([item for ll in inputX for item in ll] + ['', '<SOS>', '<EOS>'] + [str(i) for i in range(max_seq_len)]))}
    idx2word = {word2idx[w]: w for w in word2idx}
    vocabulary_size = len(word2idx)
    print(f'\033[32;1mvocabulary_size: {vocabulary_size}\033[0m')

    hidden_dim = 128
    model, encoder_model, decoder_model= build_model(hidden_dim, max_seq_len, vocabulary_size)
    plot_model(model, show_shapes=True)

    model.compile(Adam(1e-3), loss='sparse_categorical_crossentropy', loss_weights=[1., 10.], metrics=['sparse_categorical_accuracy'])

    if os.path.exists(model_path+'.index') or os.path.exists(model_path):
        print('\033[32;1mLoading Model\033[0m')
        model.load_weights(model_path)
    if training:
        trainX, validX, trainY_SOS, validY_SOS, trainY, validY, line_len, valid_line_len = train_data_preprocessing(inputX, word2idx, max_seq_len)
        print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}, trainY_SOS: {trainY_SOS.shape}, validY_SOS: {validY_SOS.shape}\033[0m')
        checkpoint = ModelCheckpoint(model_path, 'val_word_out_loss', verbose=1, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau('val_word_out_loss', 0.5, 3, verbose=1, min_lr=1e-6)
        logger = CSVLogger(model_path+'.csv', append=True)
        epochs = 10
        for epoch in range(epochs):
            print(f'\033[32;1mepoch: {epoch+1}/{epochs}\033[0m')
            ith, ith_str, word = generate_word(trainY, line_len)
            print(' '.join([idx2word[i] for i in trainX[8347]]).strip(), ith[8347, 0], idx2word[word[8347, 0]], ith[8347, 1], idx2word[word[8347, 1]])
            print(' '.join([idx2word[i] for i in trainY[8347]]).strip())
            print(' '.join([idx2word[i] for i in trainY_SOS[8347]]).strip())
            valid_ith, valid_ith_str, valid_word = generate_word(validY, valid_line_len)
           
            model.fit([trainX, trainY_SOS, ith, ith_str, word], [trainY, word], validation_data=([validX, validY_SOS, valid_ith, valid_ith_str, valid_word], [validY, valid_word]), batch_size=256, epochs=1, callbacks=[checkpoint, reduce_lr, logger])

        ith, ith_str, word = generate_word(trainY, line_len)
        valid_ith, valid_ith_str, valid_word = generate_word(validY, valid_line_len)

        print(f'Training score: {model.evaluate([trainX, trainY_SOS, ith, ith_str, word], [trainY, word], batch_size=256, verbose=0)}')
        print(f'Validaiton score: {model.evaluate([validX, validY_SOS, valid_ith, valid_ith_str, valid_word], [validY, valid_word], batch_size=256, verbose=0)}')

    if submit:
        with open('output.txt', 'w') as f, open(submit, 'r') as t:
            input_testX = [line.split() for line in t.readlines()]
            testX = [[word2idx[w] if w in word2idx else word2idx[''] for w in ll[:ll.index('<EOS>')+1]] for ll in input_testX]
            testX = pad_sequences(testX, max_seq_len, padding='post', truncating='post', value=word2idx[''])
            
            word_n = [(len(ll) - ll.index('<EOS>'))//2 for ll in input_testX]
            print(word_n[:10])
            #the index of the word in testX starts from 1, which is different from our model
            test_ith = np.array([ list(map(int, ll[-word_n[i]*2::2])) * (2 // word_n[i]) for i, ll in enumerate(input_testX)], dtype=np.int32) - 1
            test_ith_str = np.array([[word2idx[str(i[0])], word2idx[str(i[1])]] for i in test_ith], dtype=np.int32)
            test_word = np.array([list(map(lambda w: word2idx[w] if w in word2idx else word2idx[''], ll[-word_n[i]*2+1::2])) * (2 // word_n[i]) for i, ll in enumerate(input_testX)], dtype=np.int32)

            for i in range(testX.shape[0]):
                decoder_seq = decode_sequence(encoder_model, decoder_model, testX[i:i+1], test_ith[i:i+1], test_ith_str[i:i+1], test_word[i:i+1], max_seq_len, word2idx)
                print(' '.join([idx2word[idx] for idx in decoder_seq.ravel()]).strip(), file=f)
