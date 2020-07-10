# coding=utf-8

import numpy as np
from scipy.io import loadmat
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 配置参数
epoches = 100
alpha = 0.01
K = 10


# 加载mat数据
def load_data():
    datas = loadmat('ex3data1.mat')
    X = datas['X']  # 5000*20*20
    y = datas['y']
    return X, y


# 显示数据
def display_data(X, y):
    '''随机选择数据集中100行显示'''
    # cv2.imwrite('a.png',X[1].reshape(20,20))
    # plt.figure()
    # im = Image.fromarray((X[0]*255).reshape(20,20))
    # plt.imshow(im)   #把一张图片做一些数据处理，但仅仅是做处理而已，并不显示图像
    # plt.show()
    img = np.zeros((10 * 20, 10 * 20))
    index = 0
    for line in range(10):
        for row in range(10):
            img[line * 20:(line + 1) * 20, row * 20:(row + 1) * 20] = Image.fromarray(
                (X[index] * 255).reshape(20, 20).T)
            index = index + 1

    cv2.imwrite('a.png', img)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h_func(X, theta):
    z = np.dot(X, theta)
    return sigmoid(z)


# 逻辑回归向量化
def cost_func(theta, X, y):
    m = len(X)
    h = h_func(X, theta).reshape(-1, 1)
    loss = np.dot((-y).T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))
    loss = np.sum(loss) / m
    return loss


def cost_func_reg(theta, X, y, lamda):
    m = len(X)
    loss = cost_func(theta, X, y)
    reg = lamda / (2 * m) * np.dot(theta.T, theta)
    return loss + reg


def gradient(theta, X, y):
    m = len(X)
    h = h_func(X, theta)
    grad = np.dot(X.T, (h - y))
    return grad


def gradint_reg(theta, X, y, lamda):
    m = len(X)
    grad = gradient(theta, X, y)
    grad_reg = lamda / m * theta
    return grad + grad_reg


def data_preprocessing(y):
    '''one vs all'''
    y_all = ((y == 10).astype(np.int))
    for i in range(1, 10):
        # print(i)
        y_t = ((y == i).astype(np.int))
        y_all = np.append(y_all, y_t, axis=1)

    return y_all


def train(theta, X, y_all, lamda, reg=False):
    losses = []
    thetas = np.zeros([len(X[0]), K])
    for i in range(K):
        y_t = y_all[:, i].reshape(-1, 1)
        for epo in range(epoches):
            # print(i)
            if reg == False:
                loss = cost_func(theta, X, y_t)
                losses.append(loss)
                grad = gradient(theta, X, y_t)
                theta = theta - alpha * grad
            else:
                loss = cost_func_reg(theta, X, y_t, lamda)
                losses.append(loss)
                grad_reg = gradint_reg(theta, X, y_t, lamda)
                theta = theta - alpha * grad_reg
            thetas[:, i] = theta[:, 0]
    return losses, thetas


def plot_loss(loss):  # loss总是为nan
    x = np.linspace(0, epoches, epoches)
    plt.figure()
    plt.plot(x, loss, color='red', linewidth=2)
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.show()


def normal_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    std[np.where(std == 0)] = 0.00001
    x = X.copy()
    for i in range(len(mean)):
        x[:, i] = (x[:, i] - mean[i]) / std[i]
    return x


def test(theta, X, y_all):
    preds = np.zeros((len(X), K))
    for i in range(len(theta[0])):
        theta_t = theta[:, i]
        theta_t = theta_t.reshape(-1, 1)
        pred = h_func(X, theta_t)
        preds[:, i] = pred[:, 0]
    preds = preds // np.max(preds, axis=1).reshape(-1, 1)  # 不是只看阈值（0.5），而是看所有预测类别里最高的
    acc = np.sum(preds == y_all) / (len(X) * K)
    print(acc)


if __name__ == '__main__':
    X, y = load_data()
    m = len(X)
    print(X.shape, y.shape)
    #             随机显示100行数据
    # rand_index=np.random.randint(0,m,size=100)
    # rand_X=X[rand_index]
    # rand_y=y[rand_index]
    # display_data(rand_X,rand_y)
    theta = np.ones((401, 1))
    y_all = data_preprocessing(y)
    #            训练模型
    # x = normal_data(X)
    x = np.append(np.ones((m, 1)), X, axis=1)
    _, thetas = train(theta, x, y_all, 0.001, True)
    test(thetas, x, y_all)
