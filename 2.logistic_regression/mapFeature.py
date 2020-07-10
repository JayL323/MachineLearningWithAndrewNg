# coding=utf-8
import numpy as np


def map_feature(X):
    m = len(X)
    x = np.ones((m, 1))
    x_1 = X[:, 0].reshape(-1, 1)
    x_2 = X[:, 1].reshape(-1, 1)
    for i in np.arange(1, 7):
        for x1 in range(i + 1):
            x2 = i - x1
            x = np.append(x, (x_1 ** x1) * (x_2 ** x2), axis=1)
    return x
