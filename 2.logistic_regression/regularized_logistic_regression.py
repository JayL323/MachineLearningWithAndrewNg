# coding=utf-8
import numpy as np
from plotData import plot_data
from mapFeature import map_feature
from costFunctionReg import cost_function_reg, my_fminunc
from predict import predit


def load_data():
    datas = np.loadtxt('ex2data2.txt', delimiter=',')
    X = datas[:, 0:2]
    y = datas[:, 2].reshape(-1, 1)
    y = y.astype(np.int)
    return X, y


def normal_data(x):
    data = x.copy()
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)

    for i in range(len(mean)):
        data[:, i] = (data[:, i] - mean[i]) / std[i]

    return data


if __name__ == '__main__':
    X, y = load_data()
    print(X[0], y[0])
    m = len(X)
    # plot_data(X,y)
    x = map_feature(X)
    X_input = np.append(np.ones((m, 1)), X, axis=1)
    n = len(x[0])
    theta = np.ones((n, 1)).reshape(-1, 1)
    # loss=cost_function_reg(x,y,theta,1)
    # print(loss)
    ret = my_fminunc(x, y, theta, 2)
    print(ret['success'])
    print(np.round(ret['x'], 2))
    theta = ret['x'].reshape(-1, 1)
    predit(x, y, theta)
