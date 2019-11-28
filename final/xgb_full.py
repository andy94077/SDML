import xgboost as xgb
import sys, os, pickle
from joblib import Parallel, delayed
import numpy as np

import utils

def score(model, X, Y):
    return np.mean(model.predict(X) == Y)

if __name__ == "__main__":
    
    data_path = sys.argv[1]
    model_path = sys.argv[2]

    trainX, trainY = utils.load_train_data(data_path)
    trainX, validX, trainY, validY = utils.train_test_split(trainX, trainY)
    print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')

    st,ed = list(map(int,sys.argv[3:5]))
    step=int(sys.argv[5]) if len(sys.argv)>=6 else 1
    
    if os.path.exists(model_path): 
        tree = utils.load_model(model_path)
        print("Load Model")
    else:
        best_train_score, best_valid_score, min_param=0, 0, 0
        base=2
        for m in range(st,ed+1,step):
            for la in range(0,4):
                print('### max_depth=',m,'lambda=',base**la,'#'*5,file=sys.stderr, flush=True)
                tree = xgb.XGBClassifier(max_depth=m, reg_lambda=base**la, n_estimators=100, objective='multi:softmax', min_child_weight=0.8, subsample=0.8, learning_rate=0.1).fit(trainX, trainY, eval_set=[(validX, validY)], verbose=False, early_stopping_rounds=150)
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

