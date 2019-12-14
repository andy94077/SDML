import numpy as np 
import utils
import sys, os
from sklearn.metrics import confusion_matrix 
from sklearn.svm import SVC, SVR 

def clf_svm(X, Y, print_cm = True, save_model = False, weight = None, params = None):
    if X is None:
        print("No input X -> for testing performance")
        X, Y = load_data('/tmp2/BLB_final/data_2/train.csv')
    X_train, X_val, Y_train, Y_val = utils.train_test_split(X, Y, valid_ratio = 0.1)
    
    if params is None:
        params = {'C':1, 'gamma':0.9, 'kernel':'rbf'} # best_rec for all 14 dims -> C = 1, gamma = 0.9
    clf = SVC(**params)
    if weight is None:
        clf.fit(X_train, Y_train)
    else:
        weight_train, _, _, _ = utils.train_test_split(weight, np.ones(X.shape[0]))
        clf.fit(X_train, Y_train, sample_weight = weight_train)
    if save_model:
        utils.save_model(save_model, clf)
    Y_pred_val = np.round(clf.predict(X_val))
    val_acc = np.mean(Y_pred_val == Y_val)
    Y_pred_tr = np.round(clf.predict(X_train))
    train_acc = np.mean(Y_pred_tr == Y_train)

    print(f"For using {X.shape[1]} dim, and using SVC")
    print("val_acc =", val_acc)
    print("training_acc =", train_acc) 
    if print_cm == True:
        print("Confusion matrix:\n", confusion_matrix(Y_val, Y_pred_val))
    return clf

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    prob_2()
    

