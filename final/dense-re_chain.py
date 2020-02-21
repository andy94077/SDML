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
from svm import clf_svm

def build_regression_model(input_shape, name=None):
    model = Sequential([
            Dense(2048, input_shape=input_shape),
            Dropout(0.5),
            Dense(2048),
            Dropout(0.5),
            Dense(2048),
            Dropout(0.5),
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

trainX, trainY = utils.load_train_data(data_path)
word2idx = {w: i for i, w in enumerate(np.unique(trainY.ravel()))}
idx2word = {word2idx[w]: w for w in word2idx}
trainY = np.array(list(map(word2idx.get, trainY.ravel())))
distribution = np.unique(trainY, return_counts=True)[1] / trainY.shape[0]
trainY = to_categorical(trainY)
label_smoothing = 0.0
trainY = trainY * (1 - label_smoothing) + label_smoothing * distribution
trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
missing_col, valid_missing_col = trainX[:, 0], validX[:, 0]
trainX = trainX[:, 1:]
validX = validX[:, 1:]
print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')


model_f1 = build_regression_model(trainX.shape[1:], name='f1')

I = Input(trainX.shape[1:])
f1 = model_f1(I)

I_full = Concatenate()([I, f1])
x = Dense(1024, 'tanh')(I_full)
#x = Dropout(0.3)(x)
x = Dense(1024, 'tanh')(x)
#x = Dropout(0.3)(x)
x = Dense(1024, 'tanh')(x)#, kernel_regularizer=l2(1e-4))(x)
out = Dense(len(word2idx), activation='softmax', name='class', kernel_regularizer=l2(1e-3))(x)

model = Model(I, [out, f1])
#model.compile(Adam(1e-3), loss=['categorical_crossentropy', 'mse'], loss_weights=[1, 1.7], metrics=[['acc'], []])
model.compile(Adam(1e-3), loss=['mse', 'mse'], loss_weights=[1, 1.7], metrics=[['acc'], []])

if os.path.exists(model_path): 
    model.load_weights(model_path)
    print('\033[32;1mLoad Model\033[0m')

plot_model(model, 'model.jpg')
if training:
    checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 10, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping('val_loss', patience=100, restore_best_weights=True)
    logger = CSVLogger(model_path+'.csv', append=True)
    tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', batch_size=1024, update_freq='epoch')
    model.fit(trainX, [trainY, missing_col], batch_size=128, epochs=500, validation_data=(validX, [validY, valid_missing_col]), verbose=2, callbacks=[checkpoint, reduce_lr, logger, tensorboard, early_stopping])

if submit:
    testX = utils.load_test_data(submit)
    Y = np.array(list(map(idx2word.get, np.argmax(model.predict(testX, batch_size=1024)[0], axis=-1))))
    utils.submit(Y)
elif svm:
    X, Y = utils.load_train_data(data_path)
    Y = np.array(list(map(word2idx.get, Y.ravel())))
    trainX = X[:, 1:]
    _, f1 = model.predict(trainX, batch_size=1024)
    X[:, 0] = f1.ravel()
    clf_svm(X, Y, save_model=svm)
else:
    model.load_weights(model_path)
    print(f'\n\033[32;1mTraining score: {model.evaluate(trainX, [trainY, missing_col], verbose=0, batch_size=1024)}')
    print(f'Validation Score: {model.evaluate(validX, [validY, valid_missing_col], verbose=0, batch_size=1024)}\033[0m')
