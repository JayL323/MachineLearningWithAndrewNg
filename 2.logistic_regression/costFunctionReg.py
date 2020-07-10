#coding=utf-8
'''Regularized Logistic Regression Cost'''
import numpy as np
from sigmoid import sigmoid
import scipy.optimize as optimize

def func_h(X,theta):
    return np.dot(X,theta)

def cost_function_reg(theta,X,y,lamda):
    m=len(X)
    h=sigmoid(func_h(X,theta)).reshape(m,1)
    loss=np.dot((-y).T,np.log(h))-np.dot((1-y).T,np.log(1-h))
    loss=np.sum(loss)/m

    theta_t=theta[1:]
    loss=loss+lamda/(2*m)*np.dot(theta_t.T,theta_t)
    return loss

def gradient(theta,X,y,lamda):
    m=len(X)
    h=func_h(X,theta).reshape(-1,1)
    grad=np.dot(X.T,(h-y))/m
    regular=lamda/m*theta.reshape(-1,1)
    grad=grad+regular
    return grad

def gradient_func(X,y,theta,alpha,lamda):
    grad=gradient(X,y,theta,lamda)
    theta=theta-alpha*grad
    return theta

def my_fminunc(X,y,theta,lamda):
    return optimize.minimize(fun=cost_function_reg,x0=theta,args=(X,y,lamda),method='TNC',jac=gradient)
