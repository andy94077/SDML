import os, sys, argparse
import numpy as np
from tqdm import trange
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard, EarlyStopping
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
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
training = not args.no_training
submit = args.submit
predict = args.predict

trainX, trainY = utils.load_train_data(data_path)#, drop_columns=['F2', 'F7', 'F12'])
if predict:
    trainY = np.round(np.load(predict))
else:
    trainY = trainY.astype(np.float32)
label_smoothing = 0.2
trainY = trainY * (1 - label_smoothing) + label_smoothing / 2
trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
missing_col, valid_missing_col = trainX[:, [1, 6, 11]], validX[:, [1, 6, 11]]
trainX = np.delete(trainX, [1, 6, 11], axis=1)
validX = np.delete(validX, [1, 6, 11], axis=1)
print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')


I = Input(trainX.shape[1:])

x = Dense(1024, 'relu')(I)
x = Dropout(0.3)(x)
x = Dense(1024, 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, 'relu')(x)#, kernel_regularizer=l2(1e-5))(x)
#x = Dropout(0.5)(x)
x_class = Dense(1024, 'relu')(x)#, kernel_regularizer=l2(1e-3))(x)
#x_class = Dropout(0.3)(x_class)
out = Dense(1, activation='sigmoid', name='class')(x_class)

x_f2 = Dense(1024, 'relu')(x)#, kernel_regularizer=l2(1e-3))(x)
f2 = Dense(1, name='f2')(x_f2)

x_f7 = Dense(1024, 'relu')(x)#, kernel_regularizer=l2(1e-3))(x)
f7 = Dense(1, name='f7')(x_f7)

x_f12 = Dense(1024, 'relu',)(x)# kernel_regularizer=l2(1e-3))(x)
f12 = Dense(1, name='f12')(x_f12)

model = Model(I, [out, f2, f7, f12])
#model.compile(Adam(1e-3), loss=['binary_crossentropy', 'mse', 'mse', 'mse'], loss_weights=[1, 3, 3, 3], metrics=[['acc'], [], [], []])
def acc(y_true, y_pred):
    return K.round(y_true[:, 0]) == K.round(y_pred[:, 0])
model.compile(Adam(1e-3), loss=['binary_crossentropy', 'mse', 'mse', 'mse'], loss_weights=[1, 3, 3, 3], metrics=[[acc], [], [], []])

if os.path.exists(model_path): 
    model.load_weights(model_path)
    print('\033[32;1mLoad Model\033[0m')

plot_model(model, 'model.jpg')
if training:
    checkpoint = ModelCheckpoint(model_path, 'val_class_acc', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('val_class_acc', 0.5, 10, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping('val_class_acc', patience=50, restore_best_weights=True)
    logger = CSVLogger(model_path+'.csv', append=True)
    tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', batch_size=1024, update_freq='epoch')
    model.fit(trainX, [trainY, missing_col[:, 0], missing_col[:, 1], missing_col[:, 2]], batch_size=512, epochs=500, validation_data=(validX, [validY, valid_missing_col[:, 0], valid_missing_col[:, 1], valid_missing_col[:, 2]]), verbose=2, callbacks=[checkpoint, reduce_lr, early_stopping, logger, tensorboard])

if submit:
    out = tf.cast(out*2, tf.int32)
    submit_model = Model(I, out)
    utils.submit(submit_model, submit)
else:
    trainX, trainY = utils.load_train_data(data_path)
    trainY = trainY.astype(int)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
    missing_col, valid_missing_col = trainX[:, [1, 6, 11]], validX[:, [1, 6, 11]]
    trainX = np.delete(trainX, [1, 6, 11], axis=1)
    validX = np.delete(validX, [1, 6, 11], axis=1)
    print(f'\n\033[32;1mTraining score: {model.evaluate(trainX, [trainY, missing_col[:, 0], missing_col[:, 1], missing_col[:, 2]], verbose=0)}')
    print(f'Validation Score: {model.evaluate(validX, [validY, valid_missing_col[:, 0], valid_missing_col[:, 1], valid_missing_col[:, 2]], verbose=0)}\033[0m')
