import os, sys, argparse
import numpy as np
from tqdm import trange
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf

import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('model_path')
parser.add_argument('-T', '--no-training', action='store_true')
parser.add_argument('-s', '--submit')
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
training = not args.no_training
submit = args.submit

trainX, trainY = utils.load_train_data(data_path, drop_columns=['F1'])
word2idx = {w: i for i, w in enumerate(np.unique(trainY))}
idx2word = {word2idx[w]: w for w in word2idx}
trainY = np.array(list(map(word2idx.get, trainY.ravel())))
distribution = np.unique(trainY, return_counts=True)[1] / trainY.shape[0]
trainY = to_categorical(trainY)
label_smoothing = 0.0
trainY = trainY * (1 - label_smoothing) + label_smoothing * distribution
trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')

I = Input(trainX.shape[1:])
x = Dense(1024, 'tanh')(I)
#x = Dropout(0.3)(x)
x = Dense(1024, 'tanh')(x)
#x = Dropout(0.3)(x)
x = Dense(1024, 'tanh')(x)#, kernel_regularizer=l2(1e-4))(x)
#x = Dropout(0.5)(x)
#x = Dense(1024, 'tanh')(x)#, kernel_regularizer=l2(1e-3))(x)
#x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
out = Dense(len(word2idx), activation='softmax')(x)#, kernel_regularizer=l2(1e-3))(x)

model = Model(I, out)
model.compile(Adam(1e-3), loss='categorical_crossentropy', metrics=['acc'])

if os.path.exists(model_path): 
    model.load_weights(model_path)
    print('\033[32;1mLoad Model\033[0m')

plot_model(model, 'model.jpg')
if training:
    checkpoint = ModelCheckpoint(model_path, 'val_acc', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('val_acc', 0.7, 10, verbose=1, min_lr=1e-6)
    logger = CSVLogger(model_path+'.csv', append=True)
    tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', batch_size=1024, update_freq='epoch')
    model.fit(trainX, trainY, batch_size=128, epochs=100, validation_data=(validX, validY), callbacks=[checkpoint, reduce_lr, logger, tensorboard], verbose=2)


if submit:
    testX = utils.load_test_data(submit)
    Y = np.array(list(map(idx2word.get, np.argmax(model.predict(testX, batch_size=512), axis=-1))))
    utils.submit(Y)
else:
    if training:
        model.load_weights(model_path)
    print(f'\n\033[32;1mTraining score: {model.evaluate(trainX, trainY, verbose=0)}')
    print(f'Validation Score: {model.evaluate(validX, validY, verbose=0)}\033[0m')
