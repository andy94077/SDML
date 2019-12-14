import os, sys, argparse
import numpy as np
from tqdm import trange
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
import tensorflow as tf

import utils
from train_2step import clf_svm

def build_regression_model(input_shape, name=None):
    model = Sequential([
            Dense(2048, input_shape=input_shape),
            LeakyReLU(0.2),
            Dropout(0.3),
            Dense(2048),
            LeakyReLU(0.2),
            Dropout(0.3),
            Dense(1024),
            LeakyReLU(0.2),
            Dropout(0.5),
            Dense(1024),
            LeakyReLU(0.2),
            Dropout(0.5),
            Dense(1)
        ], name=name)
    return model

os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('model_path')
parser.add_argument('-T', '--no-training', action='store_true')
parser.add_argument('-s', '--submit')
parser.add_argument('--svm', nargs='?', const='.svm', default=False)
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
training = not args.no_training
submit = args.submit
svm = model_path[:-3] + args.svm if args.svm else args.svm

trainX, trainY = utils.load_train_data(data_path)#, drop_columns=['F2', 'F7', 'F12'])
trainY = trainY.astype(np.float32)
label_smoothing = 0.1
trainY = trainY * (1 - label_smoothing) + label_smoothing / 2
trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
missing_col, valid_missing_col = trainX[:, [1, 6, 11]], validX[:, [1, 6, 11]]
trainX = np.delete(trainX, [1, 6, 11], axis=1)
validX = np.delete(validX, [1, 6, 11], axis=1)
print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')


model_f2 = build_regression_model(trainX.shape[1:], name='f2')
model_f7 = build_regression_model(trainX.shape[1:], name='f7')
model_f12 = build_regression_model(trainX.shape[1:], name='f12')

I = Input(trainX.shape[1:])
f2 = model_f2(I)
f7 = model_f7(I)
f12 = model_f12(I)

I_full = Concatenate()([I, f2, f7, f12])
x = Dense(1024, 'relu')(I_full)
#x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, 'relu')(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, 'relu', kernel_regularizer=l2(5e-3))(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, 'relu', kernel_regularizer=l2(5e-3))(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3), name='class')(x)

model = Model(I, [out, f2, f7, f12])
#model.compile(Adam(1e-3), loss=['binary_crossentropy', 'mse', 'mse', 'mse'], loss_weights=[1, 3, 3, 3], metrics=[['acc'], [], [], []])#, ['mse'], ['mse'], ['mse']])
def acc(y_true, y_pred):
    return K.round(y_true[:, 0]) == K.round(y_pred[:, 0])
model.compile(Adam(1e-3), loss=['binary_crossentropy', 'mse', 'mse', 'mse'], loss_weights=[1, 3, 3, 3], metrics=[[acc], [], [], []])

if os.path.exists(model_path): 
    model.load_weights(model_path)
    print('\033[32;1mLoad Model\033[0m')

plot_model(model, 'model.jpg')
if training:
    checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 10, verbose=1, min_lr=1e-6)
    logger = CSVLogger(model_path+'.csv', append=True)
    tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', batch_size=1024, update_freq='epoch')
    model.fit(trainX, [trainY, missing_col[:, 0], missing_col[:, 1], missing_col[:, 2]], batch_size=256, epochs=100, validation_data=(validX, [validY, valid_missing_col[:, 0], valid_missing_col[:, 1], valid_missing_col[:, 2]]), verbose=2, callbacks=[checkpoint, reduce_lr, logger, tensorboard])

if submit:
    out = tf.cast(out*2, tf.int32)
    submit_model = Model(I, out)
    utils.submit(submit_model, submit)
elif svm:
    X, Y = utils.load_train_data(data_path)
    Y = Y.astype(int)
    trainX = np.delete(X, [1, 6, 11], axis=1)
    _, f2, f7, f12 = model.predict(trainX, batch_size=1024)
    X[:, 1] = f2.ravel()
    X[:, 6] = f7.ravel()
    X[:, 11] = f12.ravel()
    clf_svm(X, Y, save_model=svm)
else:
    model.load_weights(model_path)
    print(f'\n\033[32;1mTraining score: {model.evaluate(trainX, [trainY, missing_col[:, 0], missing_col[:, 1], missing_col[:, 2]], verbose=0)}')
    print(f'Validation Score: {model.evaluate(validX, [validY, valid_missing_col[:, 0], valid_missing_col[:, 1], valid_missing_col[:, 2]], verbose=0)}\033[0m')
