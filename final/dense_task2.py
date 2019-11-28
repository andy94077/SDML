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

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('model_path')
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path

trainX, trainY = utils.load_train_data(data_path)#, drop_columns=['F2', 'F7', 'F12'])
trainY = trainY.astype(int)
trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
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
    print("Load Model")

keep_training = True
if keep_training:
    checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 10, verbose=1, min_lr=1e-6)
    logger = CSVLogger(model_path+'.csv', append=True)
    tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', batch_size=1024, update_freq=512)
    model.fit(trainX, trainY, batch_size=128, epochs=50, validation_data=(validX, validY), callbacks=[checkpoint, reduce_lr, logger, tensorboard])

print(f'\n\033[32;1mTraining score: {model.evaluate(trainX, trainY, verbose=0)}')
print(f'Validation Score: {model.evaluate(validX, validY, verbose=0)}\033[0m')
