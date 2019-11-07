import os, sys, argparse
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, RepeatVector, TimeDistributed, Lambda, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.version.VERSION == '2.0.0':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

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

max_seq_len = 100
with open(data, 'r') as f:
    inputX = [line.strip() for line in f.readlines()]
word2idx = { w: i for i, w in enumerate(np.unique([item for ll in inputX for item in ll] + ['', '<SOS>', '<EOS>']))}
idx2word = {word2idx[w]: w for w in word2idx}
vocabulary_size = len(word2idx)

trainX = [[word2idx['<SOS>']]+[word2idx[w] for w in ll[5:-5]]+[word2idx['<EOS>']] for ll in inputX] # [5:-5] ignores <SOS> and <EOS>
trainX = pad_sequences(trainX, max_seq_len, padding='post', value=word2idx[''])
np.random.seed(880301)
idx = np.random.permutation(trainX.shape[0])
trainX, validX = trainX[idx[:int(trainX.shape[0]*0.9)]], trainX[idx[int(trainX.shape[0]*0.9):]]
trainY = np.concatenate([trainX[:,1:], np.full((trainX.shape[0], 1), word2idx[''])], axis=1)
validY = np.concatenate([validX[:,1:], np.full((validX.shape[0], 1), word2idx[''])], axis=1)
print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}\033[0m')

hidden_dim = 192
encoder_in = Input((max_seq_len,), dtype='int32')
x = Lambda(lambda x: K.one_hot(x, vocabulary_size))(encoder_in)
encoder_out, state = GRU(hidden_dim, return_state=True)(x)
encoder_out = RepeatVector(max_seq_len)(encoder_out)

decoder_in = Input((max_seq_len,), dtype='int32')
x = Lambda(lambda x: K.one_hot(x, vocabulary_size))(decoder_in)
x = Concatenate()([x, encoder_out])
x = GRU(hidden_dim, return_sequences=True)(x, initial_state=state)
decoder_out = Dense(vocabulary_size, activation='softmax')(x)

model = Model([encoder_in, decoder_in], decoder_out)
#model.summary()
#plot_model()

def acc(y_true, y_pred):
    return K.mean(K.all(K.equal(tf.cast(K.reshape(y_true, (-1, max_seq_len)), tf.int64), K.argmax(y_pred, axis=-1)), axis=-1))

model.compile(Adam(1e-4, clipnorm=0.5), loss='sparse_categorical_crossentropy' , metrics=[acc, 'sparse_categorical_accuracy'])

if os.path.exists(model_path+'.index') or os.path.exists(model_path):
    print('\033[32;1mLoading Model\033[0m')
    model.load_weights(model_path)
if training:
    checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 3, verbose=1, min_lr=1e-6)
    logger = CSVLogger(model_path+'.csv', append=True)
    model.fit([trainX, trainX], trainY, validation_data=([validX, validX], validY), batch_size=512, epochs=100, callbacks=[checkpoint, reduce_lr, logger])

if submit:
    with open('output.txt', 'w') as f, open(submit, 'r') as t:
        testX = [line.strip() for line in t.readlines()]
        # [5:-5] ignores <SOS> and <EOS>
        testX = [[word2idx['<SOS>']]+[word2idx[w] if w in word2idx else word2idx[''] for w in ll[5:-5]]+[word2idx['<EOS>']] for ll in testX]
        testX = pad_sequences(testX, max_seq_len, padding='post', value=word2idx[''])
        pred = np.argmax(model.predict([testX,testX], batch_size=512), axis=-1)
        for i in pred:
            f.write('<SOS> ' + ''.join([idx2word[idx] for idx in i]).strip() + '\n')
else:
    print(f'Training score: {model.evaluate([trainX, trainX], trainY, batch_size=256, verbose=0)}')
    print(f'Validaiton score: {model.evaluate([validX, validX], validY, batch_size=256, verbose=0)}')
