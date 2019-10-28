import numpy as np
import pandas as pd
import sys, os, random
import util

def f1_acc(Y_pred, Y_true):
    Y_pred = ( Y_pred > 0.5 )
    precision = np.sum((Y_pred == Y_true[:, :-1]) & (Y_true[:, :-1] == 1)) / np.sum(Y_pred)
    recall = np.sum((Y_pred == Y_true[:, :-1]) & (Y_true[:, :-1] == 1)) / np.sum(Y_true[:, :-1])
    return 2 * precision * recall / (precision + recall)

csv = sys.argv[1]
Y_pred = pd.read_csv(csv).drop('order_id', axis=1).values[:7000]

X, Y, seg = util.load_task2_trainXY('bert_dataset/wwm_uncased_L-24_H-1024_A-16/vocab.txt', 'data')
random.seed(246601)
rd_seq = random.sample(range(X.shape[0]), X.shape[0])
print(f1_acc(Y_pred[:, :3], Y_val))
