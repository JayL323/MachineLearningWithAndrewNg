#coding=utf-8
import numpy as np
from plot_data import plot_data,plot_loss

epoches=5000
alpha=0.001

def load_data():
    datas=np.loadtxt('ex1data2.txt',delimiter=',')
    return datas

def feature_normalize(X):
    '''
        多个特征时，需要进行特征归一化
        （1）计算均值
        （2）计算标准差
        （3）归一化
    '''
    x=X.copy()
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0,ddof=1)   #ddof必须
    for i in range(len(mean)):
        x[:,i]=(x[:,i]-mean[i])/std[i]
    return x,mean,std
def func_h(X,theta):
    return np.dot(X,theta)

def compute_loss(h,y):
    m=len(h)
    deta=h-y
    loss=np.dot(deta.T,deta)
    loss=sum(loss)/(2*m)
    return loss

def gradient_descent(x,y,theta,alpha):
    m=len(x)
    deta=func_h(x,theta)-y
    theta=theta-alpha/m*np.dot(x.T,deta).reshape(3,1)
    return theta

def val_data(X,y,mean,std,theta):
    # 验证数据
    x_test=X.copy()
    print(X,y)
    for i in range(len(mean)):
        x_test[i]=(x_test[i]-mean[i])/std[i]
    x_test=np.append([1],x_test)
    pred_y=func_h(x_test,theta)
    print(pred_y)

def normal_eqn(x,y):
    '''不需要归一化'''
    theta=np.linalg.inv(np.dot(x.T,x))
    theta=np.dot(theta,x.T)
    theta=np.dot(theta,y)
    return theta

if __name__ == '__main__':
    datas=load_data()
    X=datas[:,0:2].reshape(-1,2)
    y=datas[:,2].reshape(-1,1)
    m=len(X)
    x,mean,std=feature_normalize(X)
    x=np.append(np.ones((m,1)),x,axis=1)
    theta=np.zeros((3,1))

    losses=[]
    for i in range(epoches):
        h=func_h(x,theta)
        loss=compute_loss(h,y)
        losses.append(loss)
        theta=gradient_descent(x,y,theta,alpha)
    print(theta)
    val_data(X[0], y[0], mean, std,theta)
    #plot_loss(np.linspace(0,epoches,num=epoches),losses)
    print('*'*10)
    theta=normal_eqn(X, y)
    print(theta)
    pred_y=func_h(X[0],theta)
    print(pred_y)
