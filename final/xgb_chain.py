import xgboost as xgb
import sys, os, pickle
from joblib import Parallel, delayed
import numpy as np

import utils

class XGBChain():
    def __init__(self, full_model, f1_model):
        self.full_model = full_model
        self.f1_model = f1_model
    
    def predict(self, X):
        f1 = self.f1_model.predict(X).reshape(-1, 1)
        X = np.concatenate([f1, X], axis=1)
        return self.full_model.predict(X)

def score(model, X, Y):
    return np.mean(model.predict(X) == Y)

if __name__ == "__main__":
    
    data_path = sys.argv[1]
    full_model_path = sys.argv[2]
    f1_model_path = sys.argv[3]
    submit = sys.argv[4]

    trainX, trainY = utils.load_train_data(data_path)
    trainX, validX, trainY, validY = utils.train_test_split(trainX[:, 1:], trainY)
    print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')

    
    full_model = utils.load_model(full_model_path)
    f1_model = utils.load_model(f1_model_path)
    model = XGBChain(full_model, f1_model)
    
    print('\nTraining score:', score(model, trainX, trainY),file=sys.stderr)
    print('Validation Score:', score(model, validX, validY), file=sys.stderr)
    utils.submit(model, submit)

