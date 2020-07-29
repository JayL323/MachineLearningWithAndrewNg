#coding=utf-8
'''
    collaborative filtering learning algorithm
    and apply it to a dataset of movie ratings
'''

from scipy.io import loadmat
import numpy as np
#from scipy.optimize import minimize
import matplotlib.pyplot as plt

global grad
global grad_reg

def cofi_cost_func(parms,y,r,num_features=10):
    n_m,n_u=y.shape
    theta=parms[:n_u*num_features].reshape(n_u,num_features)
    x=parms[n_u*num_features:].reshape(n_m,num_features)
    #r_index=np.where(r==1)
    #t0=np.dot(x,theta.T)[r_index]
    diff=(np.dot(x,theta.T)-y)*r
    cost=np.sum(diff**2)/2

    x_grad=np.dot(diff,theta)
    theta_grad=np.dot(diff.T,x)
    global grad
    grad=np.append(np.ravel(theta_grad),np.ravel(x_grad))
    return cost

def cofi_cost_func_reg(parms,y,r,num_features,lamda):
    n_m,n_u=y.shape
    theta=parms[:n_u*num_features].reshape(n_u,num_features)
    x=parms[n_u*num_features:].reshape(n_m,num_features)
    diff=(np.dot(x,theta.T)-y)*r
    cost=np.sum(diff**2)/2
    cost_reg=lamda/2*np.sum(theta**2)+lamda/2*np.sum(x**2)
    cost_reg+=cost

    x_grad=np.dot(diff,theta)
    theta_grad=np.dot(diff.T,x)
    global grad_reg
    x_grad+=lamda*x
    theta_grad+=lamda*theta
    grad_reg = np.append(np.ravel(theta_grad), np.ravel(x_grad))
    return cost_reg

def gradient(parms,y,r,num_features=10):
    global grad
    return grad

def gradient_reg(parms,y,r,num_features,lamda):
    global grad_reg
    return grad_reg


if __name__ == '__main__':
    #--------Part 1 Movie ratings dataset--------
    #Y (a num movies × num users matrix
    #R(i; j) = 1 if user j gave a rating to movie i, and R(i; j) = 0 otherwise
    datas=loadmat('data/ex8_movies.mat')
    Y,R=datas['Y'],datas['R']
    #The i-th row of X corresponds to the feature vector x(i) for the i-th movie
    #the j-th row of Theta corresponds to one parameter vector θ(j)
    datas=loadmat('data/ex8_movieParams.mat')
    X,Theta=datas['X'],datas['Theta']              #X和Theat都不知道的情况；相互学习，协同过滤算法
    num_users,num_movies,num_features=datas['num_users'],datas['num_movies'],datas['num_features']

    #--------Part 2 Collaborative filtering learning algorithm--------
    #...当特征和数据太多时，不容易回归；应先使用PCA再进行回归
    num_users_reduce = 4
    num_movies_reduce = 5
    num_features_reduce = 3

    X_reduce = X[np.arange(num_movies_reduce), :]
    X_reduce = X_reduce[:, np.arange(num_features_reduce)]

    Theta_reduce = Theta[np.arange(num_users_reduce), :]
    Theta_reduce = Theta_reduce[:, np.arange(num_features_reduce)]

    Y_reduce = Y[np.arange(num_movies_reduce), :]
    Y_reduce = Y_reduce[:, np.arange(num_users_reduce)]

    R_reduce = R[np.arange(num_movies_reduce), :]
    R_reduce = R_reduce[:, np.arange(num_users_reduce)]

    parms=np.append(np.ravel(Theta_reduce),np.ravel(X_reduce),axis=0)
    global grad
    grad=np.zeros(parms.shape)
    cost=cofi_cost_func(parms,Y_reduce,R_reduce,num_features_reduce)
    print(cost)
    #Collaborative filtering gradient
    # ret=minimize(cofi_cost_func,parms,args=(Y_reduce,R_reduce,num_features_reduce),method='TNC', jac=gradient)
    # x=ret['x']
    #print(ret)
    cost=cofi_cost_func(parms,Y_reduce,R_reduce,num_features_reduce)
    # losses=[]
    # for i in range(300):
    #     grad=gradient(parms,Y_reduce,R_reduce,num_features_reduce)
    #     parms=parms-0.001*grad
    #     cost=cofi_cost_func(parms, Y_reduce, R_reduce,num_features_reduce)
    #     losses.append(cost)
    # plt.figure()
    # plt.plot(np.arange(0,len(losses)),losses,color='blue')
    # plt.show()
    lamda=10
    losses=[]
    cost = cofi_cost_func_reg(parms, Y_reduce, R_reduce, num_features_reduce, lamda)
    for i in range(300):
        grad=gradient_reg(parms,Y_reduce,R_reduce,num_features_reduce,lamda)
        parms=parms-0.001*grad
        cost=cofi_cost_func_reg(parms, Y_reduce, R_reduce,num_features_reduce,lamda)
        losses.append(cost)
    plt.figure()
    plt.plot(np.arange(0,len(losses)),losses,color='blue')
    plt.show()

    #--------Part 3 Learning movie recommendations--------
    theta = parms[:num_users_reduce * num_features_reduce].reshape(num_users_reduce, num_features_reduce)
    x = parms[num_users_reduce * num_features_reduce:].reshape(num_movies_reduce, num_features_reduce)
    y_pred=np.dot(x,theta.T)
    print(y_pred.shape)
