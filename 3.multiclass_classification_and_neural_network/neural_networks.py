# coding=utf-8
'''神经网络可以构造比逻辑分类复杂的多的模型，处理复杂的特征'''

from scipy.io import loadmat
import numpy as np

K=10
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

    def one_to_all_y(self,y):
        m = len(y)
        y_all = np.zeros((m, K), dtype=np.int)
        for i in range(1, K + 1):
            y_t = np.zeros((m, 1), dtype=np.int)
            y_t[np.where(y[:, 0] == i), 0] = 1
            y_all[:, i - 1] = y_t[:, 0]
        return y_all

    def cost_func(self, X, y):
        losses = 0
        m = len(X)
        # for i in range(K):
        #     h=self.forward(X)    #m*10
        #     y_k=y[:,i].reshape(-1,1)           #m*1
        #     loss=-np.dot(y_k.T,np.log(h))-np.dot((1-y_k).T,np.log(1-h))  #1*10
        #     losses+=(np.sum(loss)/m)
        losses = 0
        for i in range(m):
            cur_x, cur_y = X[i].reshape(1, -1), y[i].reshape(1, -1)  # 1*400,1*10
            pred_y = self.forward(cur_x)  # 1*10
            # loss=np.dot((-cur_y).T,np.log(pred_y))-np.dot((1-cur_y).T,np.log(1-pred_y))  #错误
            loss = np.sum((-cur_y) * np.log(pred_y)) - np.sum((1 - cur_y) * np.log(1 - pred_y))  # 当前标签和输出按位置相乘
            losses += loss
        losses = losses / m
        return losses

if __name__ == '__main__':
    theta1, theta2 = load_weights()
    X, y = load_data()
    nn = ModelNN(theta1, theta2)
    x = np.append(np.ones((len(X), 1)), X, axis=1)
    y_all=nn.one_to_all_y(y)
    loss=nn.cost_func(x,y_all)
    print('loss =',loss)
    out = nn.forward(x)
    pred = out // np.max(out, axis=1).reshape(-1, 1)
    print(pred.shape)
    acc = np.sum([1 for i in range(len(y)) if pred[i, y[i] - 1] == 1])
    print(acc / len(y))
