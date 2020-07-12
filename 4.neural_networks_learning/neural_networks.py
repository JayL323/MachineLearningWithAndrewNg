# coding=utf-8

from scipy.io import loadmat
from scipy.optimize import fmin_cg
import numpy as np
import cv2
import matplotlib.pyplot as plt

K = 10
epoches = 50
lr_rate = 0.005


def load_data():
    datas = loadmat('ex4data1.mat')
    X = datas['X']  # 5000*20*20
    y = datas['y']
    return X, y


def load_weights():
    datas = loadmat('ex4weights.mat')
    theta1 = datas['Theta1']  # 25*401
    theta2 = datas['Theta2']  # 10*26
    return theta1, theta2


class MoudeNN():
    def __init__(self, theta1, theta2, lamda, check=False):
        self.theta1 = theta1
        self.theta2 = theta2
        self.lamda = lamda
        self.check = check
        if check == True:
            self.theta1_check = theta1.copy()
            self.theta2_check = theta2.copy()

    def randInitializeWeights(self):
        '''对于各层参数随机初始化
        W = rand(L out, 1 + L in) * 2 * epsilon_init − epsilon_init;
        epsilon_init = 根号6/根号(L_in+L_out) , where Lin = sl and Lout = sl+1 是参数θ的相邻两侧单元数
        theta1 = datas['Theta1']  # 25*401
        theta2 = datas['Theta2']  # 10*26
        '''
        epsilon_init_1 = np.sqrt(6) / np.sqrt(25 + 400)  # 大约是0.12，输出的行数+输入的列数
        print(epsilon_init_1)
        self.theta1 = np.random.rand(25, 400 + 1) * 2 * epsilon_init_1 - epsilon_init_1  # 产生0-1的随机数
        epsilon_init_2 = np.sqrt(6) / np.sqrt(10 + 26)
        print(epsilon_init_2)
        self.theta2 = np.random.rand(10, 25 + 1) * 2 * epsilon_init_2 - epsilon_init_2

    def forward(self, X):
        m = len(X)
        x = np.append(np.ones((m, 1)), X, axis=1)
        if self.check == True:
            a2 = self.sigmoid(self.theta1_check, x)
            a2 = np.append(np.ones((m, 1)), a2, axis=1)
            a3 = self.sigmoid(self.theta2_check, a2)
        else:
            a2 = self.sigmoid(self.theta1, x)
            a2 = np.append(np.ones((m, 1)), a2, axis=1)
            a3 = self.sigmoid(self.theta2, a2)
        return a3

    def h_func(self, theta, X):
        return np.dot(X, theta.T)

    def sigmoid(self, theta, X):
        z = self.h_func(theta, X)
        return 1 / (1 + np.exp(-z))

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
        print(losses)
        return losses

    def cost_func_reg(self, X, y):
        losses = self.cost_func(X, y)
        reg_losses = self.lamda / (2 * m)
        reg_losses = reg_losses * (np.sum(self.theta1 * self.theta1) + np.sum(self.theta2 * self.theta2))
        return losses + reg_losses

    def sigmoid_gradient(self, z):
        # 对sigmoid函数求导
        gz = 1 / (1 + np.exp(-z))
        return gz * (1 - gz)

    def back_propagation(self, X, y, reg=False):
        '''按照行实现反向传播
        链式法则：J(Θ)对Θ求导=J(Θ)对z求导*z对Θ求导=Delta*x
        '''
        m = len(X)
        Delta1 = np.zeros((25, 401))
        Delta2 = np.zeros((K, 26))
        for i in range(m):
            a1 = X[i].reshape(1, -1)
            cur_y = y[i].reshape(1, -1)  # 1*10,theta1=25*401,theta2=10*26
            a1 = np.append(np.ones((1, 1)), a1, axis=1)  # 1*401
            z2 = self.h_func(self.theta1, a1)  # 1*25
            a2 = 1 / (1 + np.exp(-z2))  # 1*25
            a2 = np.append(np.ones((1, 1)), a2, axis=1)  # 1*26
            z3 = self.h_func(self.theta2, a2)  # 1*10
            a3 = 1 / (1 + np.exp(-z3))  # 1*10

            delta3 = a3 - cur_y  # 1*10
            delta2 = np.dot(delta3, self.theta2)[:, 1:] * self.sigmoid_gradient(z2)  # 1*25
            Delta2 = Delta2 + np.dot(delta3.T, a2)  # 10*25
            Delta1 = Delta1 + np.dot(delta2.T, a1)  # 25*401
        if reg == True:
            Delta2_reg = self.theta2
            Delta1_reg = self.theta1
            # Delta2 = np.append(np.zeros((K, 1)), Delta2, axis=1)
            Delta1, Delta2 = Delta1 / m + self.lamda / m * Delta1_reg, Delta2 / m + self.lamda / m * Delta2_reg
        else:
            # Delta2 = np.append(np.zeros((K, 1)), Delta2, axis=1)
            Delta1, Delta2 = Delta1 / m, Delta2 / m
        return Delta1, Delta2

    def compute_numerical_gradient(self, X, y):
        e = 1e-4
        # order='F',不能改变原来的值
        # prams = np.append(np.ravel(self.theta1_check, order='F'), np.ravel(self.theta2_check, order='F'))
        num_gards = None
        for index in range(2):
            if index == 0:
                prams = np.ravel(self.theta1_check)
            else:
                prams = np.ravel(self.theta2_check)
            num_grad = np.zeros(prams.shape)
            # 对于多个theta的每一个位置都要计算loss
            for i in range(prams.size):
                prams[i] = prams[i] + e
                loss_plus = self.cost_func(X, y)
                prams[i] = prams[i] - 2 * e
                loss_sub = self.cost_func(X, y)
                num_grad[i] = (loss_plus - loss_sub) / (2 * e)
                prams[i] = prams[i] + e
                if i > 10:
                    break
            if index == 0:
                num_gards = num_grad
            else:
                num_gards = np.append(num_gards, num_grad)
        return num_gards

    def train(self, X, y):
        losses = []
        for i in range(epoches):
            loss = self.cost_func(X, y)
            losses.append(loss)
            Delta1, Delta2 = self.back_propagation(X, y)
            self.theta1 = self.theta1 - lr_rate * Delta1
            self.theta2 = self.theta2 - lr_rate * Delta2  # 最好是能展开，一起更新
        return losses, self.theta1, self.theta2

    def my_fmin_cg(self, X, y):
        # params = fmin_cg(self.cost_func, fprime=self.back_propagation, \
        #                           args=(X,y))
        # return params
        pass


def plot_data(X):
    img = np.zeros((10 * 20, 10 * 20))
    for line in range(10):
        for row in range(10):
            img[line * 20:(line + 1) * 20, row * 20:(row + 1) * 20] = (X[10 * line + row] * 255).reshape(20, 20)
    cv2.imwrite('a.png', img.T)


def one_to_all_y(y):
    m = len(y)
    y_all = np.zeros((m, K), dtype=np.int)
    for i in range(1, K + 1):
        y_t = np.zeros((m, 1), dtype=np.int)
        y_t[np.where(y[:, 0] == i), 0] = 1
        y_all[:, i - 1] = y_t[:, 0]
    return y_all


def plot_loss(losses):
    x = np.linspace(0, len(losses), num=len(losses))
    plt.figure()
    plt.plot(x, losses, color='red', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.show()


if __name__ == '__main__':
    X, y = load_data()
    theta1, theta2 = load_weights()
    m = len(X)
    # 随机显示图像
    # x_rand = X[np.random.randint(0, m, size=100)]
    # plot_data(x_rand)

    # 前向传播
    nn = MoudeNN(theta1, theta2, 1)
    nn.randInitializeWeights()
    out = nn.forward(X)
    y_all = one_to_all_y(y)
    # loss = nn.cost_func_reg(X, y_all)

    # 反向传播
    # theta=np.zeros((25,10))
    # X=np.zeros((10,10))
    # grad=nn.sigmoid_gradient(theta,X)
    # loss = nn.cost_func(X, y_all)
    # #print(loss)
    # Delta1, Delta2 = nn.back_propagation(X, y_all)
    # gard = np.append(np.ravel(Delta1), np.ravel(Delta2))

    # check_grad
    # num_grad = nn.compute_numerical_gradient(X, y_all)
    # num_grad=num_grad[:10]
    # gard=gard[:10]
    # you should see a relative difference that is less than 1e-9
    # diff = np.linalg.norm(num_grad - gard) / np.linalg.norm(num_grad + gard)
    # print(diff)

    # 训练模型及验证模型
    losses, theta1_pred, theta2_pred = nn.train(X, y_all)
    print(losses)

    # Visualizing the hidden layer
