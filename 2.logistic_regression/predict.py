# coding=utf-8
''' Logistic Regression Prediction Function'''

import matplotlib.pyplot as plt
import numpy as np
from costFunction import func_h
from mapFeature import map_feature


def predit(X, y, theta):
    n = len(theta)
    pos = np.where(y[:, 0] == 1)
    neg = np.where(y[:, 0] == 0)
    plt.figure(figsize=(6, 6))
    pos_data = X[pos]
    x_pos_data = pos_data[:, 1]
    y_pos_data = pos_data[:, 2]
    plt.scatter(x_pos_data, y_pos_data, marker='x', linewidths=2, color='red', label="Admitted")

    neg_data = X[neg]
    x_neg_data = neg_data[:, 1]
    y_neg_data = neg_data[:, 2]
    plt.scatter(x_neg_data, y_neg_data, marker='*', linewidths=2, color='blue', label="Not admitted")
    if n <= 3:
        x_data = np.linspace(30, 100)
        y_data = np.linspace(30, 100)
        x_data, y_data = np.meshgrid(x_data, y_data)
        z = theta[0][0] + theta[1][0] * x_data + theta[2][0] * y_data  # 50*50
        print(z.shape)
        # h=1/1+np.exp(-z)
        # levels参数:确定轮廓线/区域的数量和位置。如果int Ñ，使用Ñ数据间隔; 即绘制n + 1个等高线。水平高度自动选择。
        # 如果是数组，则在指定的级别绘制轮廓线。值必须按递增顺序排列。
        # 分界线的理解：z>>0为1；z<<0,为0
        plt.contour(x_data, y_data, z, levels=[-1, 0, 1])
    else:
        x_data = np.linspace(-1, 1)
        y_data = np.linspace(-1, 1)
        z = np.zeros((len(x_data), len(y_data)))
        # z=theta[0][0]+theta[1][0]*x0+theta[2][0]*x1
        for i in range(len(x_data)):
            for j in range(len(y_data)):
                features = map_feature(np.append(x_data[i].reshape(-1, 1), y_data[j].reshape(-1, 1), axis=1))
                z[i, j] = np.dot(features, theta)  # 50*50
        # x_data, y_data = np.meshgrid(x_data, y_data)
        plt.contour(x_data, y_data, z, levels=[-1, 0, 1]).collections[0].set_label("Decision boundary")

    plt.xlabel('Exam1 Score')
    plt.ylabel('Exam2 Score')
    plt.show()
