#coding=utf-8

import numpy as np
import scipy.optimize as optimize
from scipy.io import loadmat
import cv2

static_grad=np.arange(0)
K=10

def randInitializeWeights():
    '''
        对于各层参数随机初始化(随机初始化多次，才能找到正确的结果)
    '''
    epsilon_init_1 = np.sqrt(6) / np.sqrt(25 + 400)  # 大约是0.12，输出的行数+输入的列数
    print(epsilon_init_1)
    theta1 = np.random.rand(25, 400 + 1) * 2 * epsilon_init_1 - epsilon_init_1  # 产生0-1的随机数
    epsilon_init_2 = np.sqrt(6) / np.sqrt(10 + 26)
    print(epsilon_init_2)
    theta2 = np.random.rand(10, 25 + 1) * 2 * epsilon_init_2 - epsilon_init_2
    nn_prams=np.append(np.ravel(theta1),np.ravel(theta2))
    return nn_prams

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_gradient(z):
    # 对sigmoid函数求导:g(z)(1-g(z))
    return sigmoid(z) * (1 - sigmoid(z))

def h_func(theta, X):
    return np.dot(X, theta.T)

def cost_func(nnprams,X, y):
    m = len(X)
    losses = 0
    theta1=np.reshape(nnprams[:25*401],(25,401))
    theta2=np.reshape(nnprams[25*401:],(10,26))
    Delta1=np.zeros((25,401))
    Delta2=np.zeros((10,26))
    for i in range(m):
        cur_x = X[i].reshape(1, -1)
        cur_y = y[i].reshape(1, -1)  # 1*10,theta1=25*401,theta2=10*26
        a1 = np.append(np.ones((1, 1)), cur_x, axis=1)  # 1*401
        z2 = h_func(theta1, a1)  # 1*25
        a2 = 1 / (1 + np.exp(-z2))  # 1*25
        a2 = np.append(np.ones((1, 1)), a2, axis=1)  # 1*26
        z3 = h_func(theta2, a2)  # 1*10
        a3 = 1 / (1 + np.exp(-z3))  # 1*10

        delta3 = a3-cur_y  # 1*10,求导后是a2-cur_y
        delta2 = np.dot(delta3, theta2)[:, 1:] * sigmoid_gradient(z2)  # 1*25
        Delta2 = Delta2 + np.dot(delta3.T, a2)  # 10*25
        Delta1 = Delta1 + np.dot(delta2.T, a1)  # 25*401

        # loss=np.dot((-cur_y).T,np.log(pred_y))-np.dot((1-cur_y).T,np.log(1-pred_y))  #错误
        loss = np.sum((-cur_y) * np.log(a3) - (1 - cur_y) * np.log(1 - a3))  # 当前标签和输出按位置相乘
        losses += loss

    Delta1, Delta2 = Delta1 / m, Delta2 / m
    global static_grad
    static_grad=np.append(np.ravel(Delta1),np.ravel(Delta2))
    losses = losses / m
    print('losses:',losses)
    return losses

def gradient(*args):
    '''按照行实现反向传播
    链式法则：J(Θ)对Θ求导=J(Θ)对z求导*z对Θ求导=Delta*x
    '''
    return static_grad

def cost_func_reg(nnprams,X, y,lamda):
    m=len(X)
    theta1=np.reshape(nnprams[:25*401],(25,401))
    theta2=np.reshape(nnprams[25*401:],(10,26))
    losses = cost_func(nnprams,X, y)
    reg_losses = lamda / (2 * m)
    reg_losses = reg_losses * (np.sum(theta1 *theta1) + np.sum(theta2 * theta2))
    return losses + reg_losses
def gradient_reg(nnprams,X, y,lamda):
    m=len(X)
    theta1=np.reshape(nnprams[:25*401],(25,401))
    theta2=np.reshape(nnprams[25*401:],(10,26))
    Delta2_reg = theta2
    Delta1_reg = theta1
    global  Delta1, Delta2
    Delta1=Delta1+lamda/m*Delta1_reg
    Delta2 = Delta2 + lamda / m * Delta2_reg
    global static_grad
    static_grad = np.append(np.ravel(Delta1), np.ravel(Delta2))
    return static_grad



def load_data():
    datas = loadmat('ex4data1.mat')
    X = datas['X']  # 5000*20*20
    y = datas['y']
    return X, y

def one_to_all_y(y):
    m = len(y)
    y_all = np.zeros((m, K), dtype=np.int)
    for i in range(1, K + 1):
        y_t = np.zeros((m, 1), dtype=np.int)
        y_t[np.where(y[:, 0] == i), 0] = 1
        y_all[:, i - 1] = y_t[:, 0]
    return y_all
def display_data(x):
    m, n = x.shape         #25*400
    lines=rows=int(np.sqrt(m))
    img=np.zeros((5*20,5*20))
    for line in range(lines):
        for row in range(rows):
            img[line*20:(line+1)*20,row*20:(row+1)*20]=np.reshape(x[5*line+row]*255,(20,20))
    cv2.imwrite('a.png',img)

if __name__ == '__main__':
    nn_prams=randInitializeWeights()
    X,y=load_data()
    y_all=one_to_all_y(y)
    static_grad=nn_prams       #必须先初始化
    #自己训练速度慢。最好是使用optimize.minimize方法，快速收敛
    ret=optimize.minimize(fun=cost_func,x0=nn_prams,method="CG", jac=gradient,args=(X,y_all),options={"maxiter": 50})
    print(ret)

    #显示隐藏层
    prams=ret['x']
    theta1=np.reshape(prams[:25*401],(25,401))
    theta2=np.reshape(prams[25*401:],(10,26))
    display_data(theta1[:,1:])    #1*400
