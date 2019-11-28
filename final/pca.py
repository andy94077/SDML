import sys, os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

train_csv_path = sys.argv[1]
model_path = sys.argv[2]

trainX, trainY = utils.load_train_data(train_csv_path)
print(f'\033[32;1mtrainX: {trainX.shape}, trainY: {trainY.shape}\033[0m')

if os.path.exists(model_path):
    model = utils.load_model(model_path)
else:
    model = PCA(n_components=2).fit(trainX)
    utils.save_model(model_path, model)

trainX2 = model.transform(trainX)
model = PCA(n_components=10).fit(trainX)
print('%.3f '*len(model.explained_variance_) % tuple(model.explained_variance_), '%.3f '*len(model.explained_variance_ratio_) % tuple(model.explained_variance_ratio_), sep='\n')

for c in np.unique(trainY):
    plt.scatter(trainX2[:, 0], trainX2[:, 1], label=c)
plt.legend()
plt.savefig('pca.jpg', dpi=300)
