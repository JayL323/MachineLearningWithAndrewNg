#coding=utf-8
'''Logistic Regression Cost Function'''

import numpy as np
from sigmoid import  sigmoid
import scipy.optimize as optimize

def func_h(x, theta):
    z = np.dot(x, theta)
    return sigmoid(z)

def cost_loss(theta,X, y):
    m=len(X)
    h=func_h(X,theta)         #m*1
    #print((-y).shape,h.shape)
    loss=np.dot((-y).T,np.log(h))-np.dot((1-y).T,np.log(1-h))  #注意是两者相减
    loss=np.sum(loss)/m
    return loss

def gradient_func(X,y,theta,alpha):
    m=len(X)
    h=func_h(X,theta).reshape(m,1)
    z=h-y
    gradient=np.dot(X.T,(h-y))/m
    theta=theta-alpha*gradient
    return theta

def gradient(theta,X,y):
    m=len(X)
    h=func_h(X,theta).reshape(100,1)
    gradient=np.dot(X.T,(h-y))/m
    return gradient

def my_fminunc(theta,X,y):
    return optimize.minimize(cost_loss,x0=theta,args=(X,y),method='TNC',jac=gradient)