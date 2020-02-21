import pickle, os
import numpy as np
import pandas as pd

def load_train_data(path, drop_columns=[]):
    df = pd.read_csv(path)
    df.drop(columns=['Id']+drop_columns, inplace=True)
    return df[df.columns[:-1]].values, df['Class'].values.astype(str)

def load_test_data(path):
    df = pd.read_csv(path)
    return df.dropna(axis=1).drop(columns=['Id']).values


def train_test_split(trainX, trainY, valid_ratio=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(trainX.shape[0])
    return trainX[idx[:-int(trainX.shape[0]*valid_ratio)]], trainX[idx[-int(trainX.shape[0]*valid_ratio):]], trainY[idx[:-int(trainY.shape[0]*valid_ratio)]], trainY[idx[-int(trainY.shape[0]*valid_ratio):]]

def load_model(path):
    '''
    Description
        Load models from given path
    Args
        path: model path
    Returns
        model
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(path, model):
    '''
    Description
        Save models to given path
    Args
        path: model path
    Returns
        None
    '''
    with open(path,'wb') as f:
        pickle.dump(model, f)

def submit(model, testX=None, output='output.csv'):
    if isinstance(testX, str):
        testX = load_test_data(testX)
    Y = model.predict(testX).ravel() if not isinstance(model, np.ndarray) else model
    np.savetxt(output, list(enumerate(Y)), '%s', delimiter=',', header='Id,Class', comments='')

