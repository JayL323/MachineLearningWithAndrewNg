#coding=utf-8

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\WinSxS\amd64_microsoft-windows-font-truetype-simsun_31bf3856ad364e35_10.0.18362.1_none_cd668f05ece74044\simsun.ttc", size=15)

epoch=10
lr_rate=0.001

def load_data():
    '''将数据分为训练集、验证集、测试集'''
    datas=loadmat('ex5data1.mat')
    #训练集
    X=datas['X']
    y=datas['y']
    #验证集
    Xval=datas['Xval']
    yval=datas['yval']
    #测试集
    Xtest=datas['Xtest']
    ytest=datas['ytest']
    return X,y,Xval,yval,Xtest,ytest

def plot_data(X,y):
    plt.figure(figsize=(6,6))
    plt.scatter(X[:,0],y[:,0],marker='*',color='red',linewidths=1)
    plt.xlabel(u'X轴',fontproperties=font_set)
    plt.ylabel(u'y轴',fontproperties=font_set)
    plt.show()
def h_func(X,theta):
    return np.dot(X,theta)

def cost_func(theta,X,y,lamda):
    '''线性回归损失函数,所有样本的loss'''
    m=len(X)
    theta=theta.reshape(-1,1)
    h=h_func(X,theta).reshape(-1,1)      #h-y的时候shape要一致，最好是先都转化成矩阵
    loss=np.sum((h-y)**2)/(2*m)
    reg=np.sum(theta[1:].reshape(-1,1)**2)*lamda/(2*m)
    return loss+reg

def gradiet_func(theta,X,y,lamda):
    m=len(X)
    theta=theta.reshape(-1,1)
    h=h_func(X,theta).reshape(-1,1)
    # grad_j0=np.sum((h-y)*X[:,0].reshape(m,1))/m
    # grad_j1=np.sum((h-y)*X[:,1].reshape(m,1))/m+lamda/m*np.sum(theta[1])
    # grad=np.array([grad_j0,grad_j1]).reshape(-1,1)
    grad_=np.dot(X.T,(h-y))/m
    grad_reg=grad_+lamda/m*theta.reshape(-1,1)
    grad_reg[0]=grad_[0]
    return grad_reg

def my_fmincg(theta,X,y,lamda):
    ret=minimize(fun=cost_func,x0=theta,args=(X,y,lamda),method='TNC',jac=gradiet_func)
    #print(ret)
    return ret

def plot_fit_line(X,y,theta_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], y[:, 0], marker='*', color='red', linewidth=2)

    fit_x=np.linspace(np.min(X),np.max(X)).reshape(-1,1)   #画拟合曲线时，需要自己定义x,y
    fit_x=np.append(np.ones((len(fit_x),1)),fit_x,axis=1)
    fit_y=h_func(fit_x,theta_pred)
    d=fit_x[:,0]
    plt.plot(fit_x[:,1],fit_y[:,0],color='blue',linewidth=2)

    plt.xlabel(u'X轴', fontproperties=font_set)
    plt.ylabel(u'y轴', fontproperties=font_set)
    plt.show()
if __name__ == '__main__':
    #--------Part 1 Visualizing the dataset--------
    X,y,Xval,yval,Xtest,ytest = load_data()
    m = len(X)
    n = len(X[0])
    theta = np.ones((n + 1, 1))
    x = np.append(np.ones((m, 1)), X, axis=1)
    #plot_data(X,y)

    #--------Part 2 Regularized linear regression cost function--------
    # lamda=1
    # loss=cost_func(theta,x,y,lamda)
    # print(loss)

    #--------Part 3 Regularized linear regression gradient--------
    # grad = gradiet(theta,x,y,lamda)
    # print(grad)

    #--------Part 4 Fitting linear regression--------
    lamda=0
    theta=np.zeros((n+1,1))
    # theta[:,0]=0.5
    ret=my_fmincg(theta,x,y,lamda)
    theta_pred=ret['x'].reshape(-1,1)
    plot_fit_line(X,y,theta_pred)
