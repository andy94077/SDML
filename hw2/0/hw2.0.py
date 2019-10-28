import os, sys
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, RepeatVector, TimeDistributed, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

data = sys.argv[1]
model_path = sys.argv[2]
max_seq_len = 32
with open(data, 'r') as f:
    inputX = [line.split() for line in f.readlines()]
letter2idx = { w: i for i, w in enumerate(np.unique([item for ll in inputX for item in ll] + ['']))}#{chr(i+32): i for i in range(95)}
idx2word = {letter2idx[w]: w for w in letter2idx}

trainX = [[letter2idx[w] for w in ll] for ll in inputX]
trainX = pad_sequences(trainX, max_seq_len, padding='post', value=letter2idx[''])
np.random.seed(880301)
idx = np.random.permutation(trainX.shape[0])
trainX, validX = trainX[idx[:int(trainX.shape[0]*0.9)]], trainX[idx[int(trainX.shape[0]*0.9):]]
print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}\033[0m')

hidden_dim = 256
vocabulary_size = len(letter2idx)
encoder_in = Input((max_seq_len,), dtype='int32')
x = Lambda(lambda x: K.one_hot(x, vocabulary_size))(encoder_in)
encoder_out, state = GRU(hidden_dim, return_state=True)(x)
encoder_out = RepeatVector(max_seq_len)(encoder_out)

x = GRU(hidden_dim, return_sequences=True)(encoder_out, initial_state=state)
decoder_out = TimeDistributed(Dense(vocabulary_size, activation='softmax'))(x)

model = Model(encoder_in, decoder_out)

def acc(y_true, y_pred):
    return K.mean(K.all(tf.cast(y_true, tf.int64) == K.argmax(y_pred, axis=-1), axis=-1)) 

model.compile(Adam(1e-3), loss=lambda y_true, y_pred: categorical_crossentropy(K.one_hot(tf.cast(y_true, tf.int32), vocabulary_size), y_pred), metrics=[acc])

training = True
if os.path.exists(model_path+'.index') or os.path.exists(model_path):
    model.load_weights(model_path)
    print('\033[32;1mModel loaded\033[0m')
if training:
    checkpoint = ModelCheckpoint(model_path, 'val_acc', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('val_acc', 0.5, 3, verbose=1, min_lr=1e-6)
    model.fit(trainX, trainX, validation_data=(validX, validX), batch_size=128, epochs=40, callbacks=[checkpoint, reduce_lr])

submit = False
if submit:
    test_file = sys.argv[3]
    with open('output.txt', 'w') as f, open(test_file, 'r') as t:
        testX = [line.split() for line in t.readlines()]
        testX = [[letter2idx[w] if w in letter2idx else letter2idx[''] for w in ll] for ll in testX]
        testX = pad_sequences(testX, max_seq_len, padding='post', value=letter2idx[''])
        pred = np.argmax(model.predict(testX), axis=-1)
        for i in pred:
            f.write(' '.join([idx2word[idx] for idx in i]).strip() + '\n')
else:
    print(f'Training score: {model.evaluate(trainX, trainX, batch_size=256, verbose=0)}')
    print(f'Validaiton score: {model.evaluate(validX, validX, batch_size=256, verbose=0)}')
