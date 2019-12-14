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
parser.add_argument('-p', '--predict')
parser.add_argument('-t', '--additional-training', nargs=2)
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
training = not args.no_training
submit = args.submit
predict = args.predict
additional_training = args.additional_training

trainX, trainY = utils.load_train_data(data_path)#, drop_columns=['F2', 'F7', 'F12'])
trainY = trainY.astype(int)
trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
if additional_training:
    trainX2, trainY2 = np.load(additional_training[0]), np.load(additional_training[1])
    trainX, trainY = np.concatenate([trainX, trainX2], axis=0), np.concatenate([trainY, trainY2.ravel()], axis=0)
print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')

I = Input(trainX.shape[1:])
x = Dense(1024, 'relu')(I)
#x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, 'relu')(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, 'relu', kernel_regularizer=l2(5e-3))(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, 'relu', kernel_regularizer=l2(1e-3))(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))(x)

model = Model(I, out)
model.compile(Adam(1e-3), loss='binary_crossentropy', metrics=['acc'])

if os.path.exists(model_path): 
    model.load_weights(model_path)
    print('\033[32;1mLoad Model\033[0m')

if training:
    checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 10, verbose=1, min_lr=1e-6)
    logger = CSVLogger(model_path+'.csv', append=True)
    tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', batch_size=1024, update_freq='epoch')
    model.fit(trainX, trainY, batch_size=512, epochs=50, validation_data=(validX, validY), verbose=2, callbacks=[checkpoint, reduce_lr, logger, tensorboard])

if submit:
    out = tf.cast(out*2, tf.int32)
    submit_model = Model(I, out)
    utils.submit(submit_model, submit)
elif predict:
    trainX, trainY = utils.load_train_data(data_path)#, drop_columns=['F2', 'F7', 'F12'])
    predY = model.predict(trainX, batch_size=1024).ravel()
    if predict[-4:] != '.npy':
        predict += '.npy'
    np.save(predict, predY)
else:
    if training:
        model.load_weights(model_path)
    trainX, trainY = utils.load_train_data(data_path)#, drop_columns=['F2', 'F7', 'F12'])
    trainY = trainY.astype(int)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
    print(f'\n\033[32;1mTraining score: {model.evaluate(trainX, trainY, verbose=0)}')
    print(f'Validation Score: {model.evaluate(validX, validY, verbose=0)}\033[0m')
