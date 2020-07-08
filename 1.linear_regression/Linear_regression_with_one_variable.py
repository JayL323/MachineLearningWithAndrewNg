# coding=utf-8
'''将所有的向量操作都转化成矩阵操作'''
from plot_data import plot_data, plot_loss
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3D
import matplotlib.pyplot as plt

epoches = 1500
alpha = 0.001


def load_data(plot=False):
    datas = np.loadtxt('ex1data1.txt', delimiter=',')  # np自己的加载txt的方法
    if plot == True:
        plot_data(datas[:, 0], datas[:, 1])
    return datas


def func_h(x, theta):
    return np.dot(x, theta)


def compute_cost(h, y):
    m = len(h)
    err = (h - y) ** 2
    loss = sum(err) / (2 * m)
    return loss


def gradient_descent(x, y, theta, alpha):
    m = len(x)
    deta = func_h(x, theta) - y
    theta_temp = np.sum(np.dot(x.T, deta), axis=1).reshape(2, 1)
    theta = theta - alpha / m * theta_temp
    return theta


def plot_surf(X,y):
    '''画出损失函数关于theta0和theta1的曲线'''
    theta0 = np.linspace(-10, 10,20)
    theta1 = np.linspace(-1, 4,20)
    J_val=np.zeros((len(theta0),len(theta1)))
    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_val=J_val.T
    for i in range(len(theta0)):
        for j in range(len(theta1)):
            h=func_h(X,np.array([[theta0[i]],[theta1[j]]]))
            J_val[i][j]=compute_cost(h,y)
    surf_figure=plt.figure()
    plt.subplot(111)
    ax=Axes3D(surf_figure)
    theta0,theta1=np.meshgrid(theta0,theta1)
    ax.plot_surface(theta0,theta1,J_val,cmap='jet')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    #plt.show()
    plt.savefig('surf.png')
    plt.close()

def plot_contour(X,y):
    '''画出J关于theta0,theta1的等高线'''
    theta0 = np.linspace(-10, 10,20)
    theta1 = np.linspace(-1, 4,20)
    J_val=np.zeros((len(theta0),len(theta1)))
    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_val=J_val.T
    for i in range(len(theta0)):
        for j in range(len(theta1)):
            h=func_h(X,np.array([[theta0[i]],[theta1[j]]]))
            J_val[i][j]=compute_cost(h,y)

    plt.figure()

    #输出Jval=np.logspace(-2, 3, 20)的线    等比数列，base=10
    plt.contour(theta0,theta1,J_val,np.logspace(-2, 3, 20))
    plt.plot(theta0,theta1,'rx',markersize='1')
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.show()

if __name__ == '__main__':
    datas = load_data()
    X = datas[:, 0]
    y = datas[:, 1]
    m = len(datas)

    X = X.reshape((m, 1))  # 将所有的向量都转化成矩阵操作
    y = y.reshape((m, 1))
    X = np.append(np.ones((m, 1)), X, axis=1)  # 添加列,矩阵操作，两者都是矩阵
    theta = np.zeros((2, 1))
    theta = np.array([1, 7]).reshape(2, 1)
    loss_list = []
    for i in range(epoches):
        h = func_h(X, theta)
        loss = compute_cost(h, y)
        loss_list.append(loss)
        theta = gradient_descent(X, y, theta, alpha)
    pred_y = func_h(X, theta).tolist()
    print(theta)
    # 可视化结果，这两项必不可少（1.和原数据对比 2.loss可视化）
    #plot_data(datas[:, 0], datas[:, 1], pred_y)
    #plot_loss(np.linspace(0, epoches, num=epoches), loss_list)
    plot_contour(X,y)
