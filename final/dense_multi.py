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
parser.add_argument('--svm', nargs='?', const='.svm', default=False)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
training = not args.no_training
submit = args.submit
predict = args.predict
additional_training = args.additional_training
svm = model_path[:-3] + args.svm if args.svm else args.svm
seed = args.seed if args.seed is not None else np.random.randint(0, 2147483647)
if os.path.exists(model_path+'.seed'):
    seed = int(np.loadtxt(model_path+'.seed'))
else:
    np.savetxt(model_path+'.seed', [seed])

trainX, trainY = utils.load_train_data(data_path)#, drop_columns=['F2', 'F7', 'F12'])
if predict:
    trainY = np.round(np.load(predict))
else:
    trainY = trainY.astype(np.float32)
distribution = np.unique(trainY, return_counts=True)[1][0] / trainY.shape[0]
label_smoothing = 0.2
trainY = trainY * (1 - label_smoothing) + label_smoothing * distribution
trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, seed=seed)
if additional_training:
    trainX2, trainY2 = np.load(additional_training[0])[:trainX.shape[0]], np.load(additional_training[1])[:trainY.shape[0]]
    trainX, trainY = np.concatenate([trainX, trainX2], axis=0), np.concatenate([trainY, trainY2.ravel()], axis=0)
missing_col, valid_missing_col = trainX[:, [1, 6, 11]], validX[:, [1, 6, 11]]
trainX = np.delete(trainX, [1, 6, 11], axis=1)
validX = np.delete(validX, [1, 6, 11], axis=1)
print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}, seed: {seed}\033[0m')


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

x_f12 = Dense(1024, 'relu')(x)#, kernel_regularizer=l2(1e-3))(x)
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
    checkpoint = ModelCheckpoint(model_path, 'val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau('val_loss', 0.5, 10, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping('val_loss', patience=50, restore_best_weights=True)
    logger = CSVLogger(model_path+'.csv', append=True)
    tensorboard = TensorBoard(model_path[:model_path.rfind('.')]+'_logs', batch_size=1024, update_freq='epoch')
    model.fit(trainX, [trainY, missing_col[:, 0], missing_col[:, 1], missing_col[:, 2]], batch_size=512, epochs=500, validation_data=(validX, [validY, valid_missing_col[:, 0], valid_missing_col[:, 1], valid_missing_col[:, 2]]), verbose=2, callbacks=[checkpoint, reduce_lr, early_stopping, logger, tensorboard])

if submit:
    if svm:
        testX = utils.load_test_data(submit)
        clf = utils.load_model(svm)
        _, f2, f7, f12 = model.predict(testX, batch_size=1024)
        testX = np.concatenate([testX[:, 0:1], f2, testX[:, 1:5], f7, testX[:, 5:9], f12, testX[:, 9:]], axis=1)
        Y = clf.predict(testX)
        utils.submit(Y)
    else:
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
    clf_svm(X, Y, seed, save_model=svm)
else:
    trainX, trainY = utils.load_train_data(data_path)
    trainY = trainY.astype(int)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY, seed=seed)
    missing_col, valid_missing_col = trainX[:, [1, 6, 11]], validX[:, [1, 6, 11]]
    trainX = np.delete(trainX, [1, 6, 11], axis=1)
    validX = np.delete(validX, [1, 6, 11], axis=1)
    print(f'\n\033[32;1mTraining score: {model.evaluate(trainX, [trainY, missing_col[:, 0], missing_col[:, 1], missing_col[:, 2]], verbose=0)}')
    print(f'Validation Score: {model.evaluate(validX, [validY, valid_missing_col[:, 0], valid_missing_col[:, 1], valid_missing_col[:, 2]], verbose=0)}\033[0m')
