#coding=utf-8
'''多项式回归，可以改善高偏差'''

import numpy as np
from regularized_linear_regression import load_data,cost_func,my_fmincg,h_func
import matplotlib.pyplot as plt
from bias_variance import plt_learning_curve

def poly_features(X,p):
    '''构造多项式特征'''
    x=X.copy()
    for i in range(2,p+1):
        x=np.append(x,X**i,axis=1)
    return x

def featureNormalize(X):
    '''多项式特征归一化'''
    x=X.copy()
    mu,sigma=np.mean(X,axis=0),np.std(X,axis=0,ddof=1)
    for i in range(len(mu)):
        x[:,i]=(x[:,i]-mu[i])/sigma[i]
    return x,mu,sigma

def plot_fit_line(X,y,theta_pred,mu,sigma,lamda):
    plt.figure()
    plt.scatter(X,y,marker='o',color='red',linewidths=2)
    x=np.linspace(np.min(X),np.max(X),num=1000).reshape(-1,1)
    x_poly_features=poly_features(x,8)
    for i in range(len(mu)):
        x_poly_features[:,i]=(x_poly_features[:,i]-mu[i])/sigma[i]
    x_poly_features=np.append(np.ones((len(x),1)),x_poly_features,axis=1)
    y_pred=h_func(x_poly_features,theta_pred)
    plt.plot(x,y_pred,color='blue')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('lambda={}'.format(lamda))
    plt.show()

def learning_polynomial_regression(x_norm,y,xval_norm,yval,lamda=0):
    k=len(X)
    train_loss=np.zeros(k)
    val_loss=np.zeros(k)
    for i in range(k):
        theta=np.zeros((len(x_norm[0]),1))
        ret=my_fmincg(theta,x_norm[0:i+1],y[0:i+1],lamda)
        theta_pred=ret['x'].reshape(theta.shape)

        train_loss[i]=cost_func(theta_pred,x_norm[0:i+1],y[0:i+1],lamda)
        val_loss[i]=cost_func(theta_pred,xval_norm,yval,lamda)
    #print(ret['success'])
    plt_learning_curve(train_loss,val_loss,lamda)
    plot_fit_line(X,y,theta_pred,mu,sigma,lamda)

#不同的x轴，y轴，最好掉用不同的函数
def plt_diff_lambda(train_losses,val_losses,lambda_vec):
    plt.figure()
    x=lambda_vec
    plt.plot(x,train_losses,color='red',linewidth=2,label='train_losses')
    plt.plot(x,val_losses,color='blue',linewidth=2,label='val_losses')
    plt.xlabel('lambda')
    plt.ylabel('loss')
    plt.legend(['train', 'cv'])
    plt.show()

def validation_curve(x_norm,y,xval_norm,yval,lambda_vec):
    '''使用验证集自动选择适合的λ'''
    train_loss=np.zeros(lambda_vec.shape)
    val_loss=np.zeros(lambda_vec.shape)
    for i in range(len(lambda_vec)):
        theta=np.zeros((len(x_norm[0]),1))
        ret=my_fmincg(theta,x_norm,y,lambda_vec[i])
        #print(ret['x'])
        theta_pred=ret['x'].reshape(theta.shape)
        train_loss[i]=cost_func(theta_pred,x_norm,y,0)
        val_loss[i]=cost_func(theta_pred,xval_norm,yval,0)
    #print(train_loss,val_loss)
    plt_diff_lambda(train_loss,val_loss,lambda_vec)

if __name__ =='__main__':
    #--------Part 1 Learning Polynomial Regression--------
    X, y, Xval, yval, Xtest, ytest = load_data()
    p=8             #自由度是8
    x=poly_features(X,p)
    x_norm,mu,sigma=featureNormalize(x)
    x_norm=np.append(np.ones((len(x_norm),1)),x_norm,axis=1)

    xval_norm=xval=poly_features(Xval,p)
    for i in range(len(mu)):
        xval_norm[:,i]=(xval_norm[:,i]-mu[i])/sigma[i]
    xval_norm = np.append(np.ones((len(xval_norm), 1)), xval_norm, axis=1)

    xtest_norm=xtest=poly_features(Xtest,p)
    for i in range(len(mu)):
        xtest_norm[:,i]=(xtest_norm[:,i]-mu[i])/sigma[i]
    xtest_norm=np.append(np.ones((len(xtest_norm),1)),xtest_norm,axis=1)
    #learning_polynomial_regression(x_norm, y, xval_norm, yval)


    #--------Part 2 Adjusting the regularization parameter--------
    # lamdas=[1,100]
    # for lamda in lamdas:
    #     learning_polynomial_regression(x_norm, y, xval_norm, yval,lamda)

    #--------Part 3 Selecting λ using a cross validation set--------
    # lambda_vec = np.array(([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]))
    # validation_curve(x_norm,y,xval_norm,yval,np.array(lambda_vec))

    #--------Part 4 Computing test set error--------
    #经过上一个步骤的验证，发现lambda=3是最好的选择
    theta=np.zeros((len(xtest_norm[0]),1))
    ret=my_fmincg(theta,x_norm,y,lamda=3)
    theta_pred=ret['x']
    loss=cost_func(theta_pred,xtest_norm,ytest,0)   #测试误差不加λ
    print(loss)

    #--------Part 5 Plotting learning curves with randomly selected examples--------
    m=len(x_norm)
    train_losses=np.zeros(m)
    val_losses=np.zeros(m)
    epoches=50
    for i in range(m):
        train_loss=0
        val_loss=0
        for epoch in range(epoches):
            theta=np.zeros((len(x_norm[0]),1))
            x_rand=np.random.randint(0,len(x_norm),size=i+1)
            x_,y_=x_norm[x_rand],y[x_rand]
            ret=my_fmincg(theta,x_,y_,lamda=0.01)
            theta_pred=ret['x'].reshape(-1,1)
            train_loss+=cost_func(theta_pred,x_,y_,0)

            x_rand = np.random.randint(0, len(xval_norm), size=i+1)
            xval_,yval_=xval_norm[x_rand],yval[x_rand]
            val_loss+=cost_func(theta_pred,xval_,yval_,0)
        train_losses[i]=train_loss/epoches
        val_losses[i]=val_loss/epoches
    plt_learning_curve(train_losses,val_losses,lamda=0.01)
