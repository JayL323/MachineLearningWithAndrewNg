# coding=utf-8

import numpy as np
from plotData import plot_data
from costFunction import func_h, cost_loss, gradient_func, my_fminunc
from predict import predit

epoches = 500
alpha = 0.01


def load_data():
    datas = np.loadtxt('ex2data1.txt', delimiter=',')
    X = datas[:, 0:2].reshape(-1, 2)
    y = datas[:, 2].reshape(-1, 1)
    y = y.astype(np.int)  # 转化数据类型
    return X, y


if __name__ == '__main__':
    X, y = load_data()
    m = len(X)
    print(X[0], y[0])
    # plot_data(X,y)
    x = np.append(np.ones((m, 1)), X, axis=1)
    theta = np.zeros((3, 1))
    pred_y = func_h(x, theta)
    loss = cost_loss(theta, x, y)
    ret = my_fminunc(theta, x, y)
    print(ret['x'])
    theta = ret['x'].reshape(3, 1)
    # pred_y = func_h(x, theta)
    # print(np.round(pred_y[0:10], 2))
    predit(x, y, theta)
