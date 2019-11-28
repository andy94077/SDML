import numpy as np
import sys, os, random
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import pickle, os, sys

import utils

def TSNE_and_draw(vec):
    tsne = TSNE(n_jobs = os.cpu_count(), perplexity=50.)
    embedding = tsne.fit_transform(vec)
    return embedding 

def plot_result(embedding, Y, file_path):
    for c in np.unique(Y):
        plt.scatter(embedding[:, 0], embedding[:, 1], label=c)
    plt.legend()
    plt.savefig(file_path) 


if __name__ == '__main__':
    X, Y = utils.load_train_data(sys.argv[1])

    embedding = TSNE_and_draw(X) # place accurate in the beginning
    print('finishing TSNE')
    plot_result(embedding, Y, 'tsne.jpg') 

