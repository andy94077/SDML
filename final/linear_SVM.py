import os, sys, pickle
import numpy as np
from sklearn.svm import SVC

import utils

data_path = sys.argv[1]
model_path = sys.argv[2]

trainX, trainY = utils.load_train_data(data_path)
trainX, validX, trainY, validY = utils.train_test_split(trainX[:, 1:], trainY)
print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')

if os.path.exists(model_path):
    model = utils.load_model(model_path)
else:
    best_valid_score = 0
    for c in range(-2, 2):
        print(f'c: {2**c}')
        model = SVC(C=2**c, kernel='linear').fit(trainX, trainY)
        valid_score = model.score(validX, validY)
        print(f'train / valid: {model.score(trainX, trainY)} / {valid_score}')

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_model = model
            utils.save_model(model_path, best_model)

print(f'Training score: {best_model.score(trainX, trainY)}')
print(f'Validation score: {best_model.score(validX, validY)}')
