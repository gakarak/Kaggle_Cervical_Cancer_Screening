import simple_classification as utils

import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE


if __name__ == '__main__':
    wdir = '../../dataset/train-x512-processed'
    train_idx = 'idx-mask.txt'
    valid_idx = 'idx-mask.txt'

    train = utils.readCervixes(wdir, train_idx)
    valid = utils.readCervixes(wdir, valid_idx)

    X = []
    y = []
    for t in train:
        X.append(utils.describeWithLbp(t[0], 8, 24, t[2]))
        y.append(int(t[1]))
    target_names = ["Type 1", "Type 2", "Type 3"]
    X = np.array(X)
    y = np.array(y)

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    tsne = TSNE(n_components=2, n_iter=5000)
    X_r3 = tsne.fit_transform(X, y)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [1, 2, 3], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.grid(True)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Cervix dataset')

    plt.figure()
    for color, i, target_name in zip(colors, [1, 2, 3], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.grid(True)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Cervix dataset')

    plt.figure()
    for color, i, target_name in zip(colors, [1, 2, 3], target_names):
        plt.scatter(X_r3[y == i, 0], X_r3[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.grid(True)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('t-SNE of Cervix dataset')

    plt.show()