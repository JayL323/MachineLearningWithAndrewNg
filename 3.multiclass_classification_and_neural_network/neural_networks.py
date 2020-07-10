# coding=utf-8
'''神经网络可以构造比逻辑分类复杂的多的模型，处理复杂的特征'''

from scipy.io import loadmat
import numpy as np


def load_weights():
    datas = loadmat('ex3weights.mat')
    theta1 = datas['Theta1']  # 25*401
    theta2 = datas['Theta2']  # 10*26
    return theta1, theta2


def load_data():
    datas = loadmat('ex3data1.mat')
    X = datas['X']  # 5000*20*20
    y = datas['y']
    return X, y


class ModelNN():
    def __init__(self, theta1, theta2):
        self.theta1 = theta1
        self.theta2 = theta2

    def forward(self, x):
        a2 = self.sigmoid(self.theta1, x)  # m*25
        a2 = np.append(np.ones((len(a2), 1)), a2, axis=1)  # m*26
        a3 = self.sigmoid(self.theta2, a2)  # m*10
        return a3

    def sigmoid(self, theta, X):
        z = self.h_func(theta, X)
        return 1 / (1 + np.exp(-z))

    def h_func(self, theta, X):
        return np.dot(X, theta.T)


if __name__ == '__main__':
    theta1, theta2 = load_weights()
    X, y = load_data()
    nn = ModelNN(theta1, theta2)
    x = np.append(np.ones((len(X), 1)), X, axis=1)
    out = nn.forward(x)
    pred = out // np.max(out, axis=1).reshape(-1, 1)
    print(pred.shape)
    acc = np.sum([1 for i in range(len(y)) if pred[i, y[i] - 1] == 1])
    print(acc / len(y))
