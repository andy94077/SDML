import xgboost as xgb
import sys, os, pickle
from joblib import Parallel, delayed
import numpy as np
import argparse

import utils

def score(model, X, Y):
    return np.mean(model.predict(X) == Y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('model_path')
    parser.add_argument('task', type=int)
    parser.add_argument('st', type=int)
    parser.add_argument('ed', type=int)
    parser.add_argument('-s', '--step', default=1)
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    task = args.task
    st = args.st
    ed = args.ed
    step = args.step

    trainX, trainY = utils.load_train_data(data_path)
    trainX, validX, trainY, validY = utils.train_test_split(trainX[:, 1:] if task == 1 else trainX[:, [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]], trainY)
    print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')

    if os.path.exists(model_path): 
        tree = utils.load_model(model_path)
        print("Load Model")
    else:
        best_train_score, best_valid_score, min_param=0, 0, 0
        base=2
        for m in range(st,ed+1,step):
            r = range(0, 4) if task == 1 else range(3, 7)
            for la in r:
                print('### max_depth=',m,'lambda=',base**la,'#'*5,file=sys.stderr, flush=True)
                if task == 1:
                    tree = xgb.XGBClassifier(max_depth=m, reg_lambda=base**la, n_estimators=100, objective='multi:softmax', min_child_weight=0.8, subsample=0.8, learning_rate=0.1).fit(trainX, trainY, eval_set=[(validX, validY)], verbose=False, early_stopping_rounds=150)
                else:
                    tree = xgb.XGBClassifier(max_depth=m, reg_lambda=base**la, n_estimators=200, objective='binary:logistic', min_child_weight=2, subsample=0.5, learning_rate=0.1, gamma=0.3).fit(trainX, trainY, eval_set=[(validX, validY)], verbose=False, early_stopping_rounds=150)

                train_score = score(tree, trainX, trainY)
                valid_score = score(tree, validX,validY)
                print(f'train / valid: {train_score} / {valid_score}', flush=True)
                if valid_score > best_valid_score:
                    print(f'\nval_acc improved from \033[31;1m{best_valid_score}\033[0m to \033[32;1m{valid_score}\033[0m, saving model to {model_path} ...\n', flush=True)
                    best_train_score, best_valid_score = train_score, valid_score
                    min_param = (m, base**la)
                    utils.save_model(model_path, tree)
    
    print(f'\n\033[32;1mTraining score: {best_train_score}',file=sys.stderr)
    print(f'Validation Score: {best_valid_score}\033[0m', file=sys.stderr)
    print('best tree ', min_param, file=sys.stderr)

