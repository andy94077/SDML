import xgboost as xgb
import sys, os, pickle
from joblib import Parallel, delayed
import numpy as np

import utils

class XGBoost():
    @staticmethod
    def get_tree(X, Y, evals=(), early_stopping_rounds=None, xgb_model=None, **kwargs):
        dtrain = xgb.DMatrix(X, label=Y, nthread=-1)
        tree = xgb.train(kwargs, dtrain, num_boost_round=512, evals=evals, early_stopping_rounds=early_stopping_rounds, xgb_model=xgb_model)
        return tree

    def fit(self, trainX, trainY, *args, validX=None, validY=None, **kwargs):
        evals=()
        if validX is not None and validY is not None:
            valid=xgb.DMatrix(validX, label=validY, nthread=-1)
            evals=[(valid, 'eval')]
        self.tree = XGBoost.get_tree(trainX, trainY, evals=evals, **kwargs)
        return self

    def predict(self, X, *args):
        dtrain = xgb.DMatrix(X,nthread=-1)
        return self.tree.predict(dtrain)

def score(model, X, Y):
    return np.mean((model.predict(X).ravel() - Y.ravel())**2)

if __name__ == "__main__":
    
    data_path = sys.argv[1]
    model_path = sys.argv[2]

    trainX, trainY = utils.load_train_data(data_path)
    trainX, validX, trainY, validY = utils.train_test_split(trainX[:, 1:], trainX[:, 0])
    print(f'\033[32;1mtrainX: {trainX.shape}, validX: {validX.shape}, trainY: {trainY.shape}, validY: {validY.shape}\033[0m')

    st,ed = list(map(int,sys.argv[3:5]))
    step=int(sys.argv[5]) if len(sys.argv)>=6 else 1
    
    best_train_loss, best_valid_loss, min_param = np.inf, np.inf, 0
    keep_training = True
    if os.path.exists(model_path): 
        tree = utils.load_model(model_path)
        print("Load Model")
        best_train_loss, best_valid_loss = score(tree, trainX, trainY), score(tree, validX, validY)
    if keep_training:
        base=2
        for m in range(st,ed+1,step):
            for la in range(0,4):
                print('### max_depth=',m,'lambda=',base**la,'#'*5,file=sys.stderr, flush=True)
                tree = xgb.XGBRegressor(max_depth=m, reg_lambda=base**la, n_estimators=100, objective='reg:squarederror', min_child_weight=0.8, subsample=0.8, learning_rate=0.1).fit(trainX, trainY, eval_set=[(validX, validY)], verbose=False, early_stopping_rounds=150)
                train_score = score(tree, trainX, trainY)
                valid_score = score(tree, validX,validY)
                print(f'train / valid: {train_score} / {valid_score}', flush=True)
                if valid_score < best_valid_loss:
                    print(f'\nval_loss improved from \033[31;1m{best_valid_loss}\033[0m to \033[32;1m{valid_score}\033[0m, saving model to {model_path} ...\n', flush=True)
                    best_train_loss, best_valid_loss = train_score, valid_score
                    min_param = (m, base**la)
                    utils.save_model(model_path, tree)
    
    print(f'\n\033[32;1mTraining score: {best_train_loss}',file=sys.stderr)
    print(f'Validation Score: {best_valid_loss}\033[0m', file=sys.stderr)
    print('best tree ', min_param, file=sys.stderr)

