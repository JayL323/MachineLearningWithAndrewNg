#coding=utf-8
'''
    plot training and test errors on a learning curve to diagnose bias-variance problems
    开始：高偏差（训练误差极大和验证误差极小），可以随着数据集的增大而改善
    高方差：训练集和验证集都趋于平缓，但两者最终相差很大
'''
from regularized_linear_regression import load_data,cost_func,gradiet_func,my_fmincg
import numpy as np
import matplotlib.pyplot as plt

def plt_learning_curve(train_losses,val_losses,lamda=0):
    plt.figure()
    k=len(train_losses)
    x=np.linspace(0,k,num=k)
    plt.plot(x,train_losses,color='red',linewidth=2,label='train_losses')
    plt.plot(x,val_losses,color='blue',linewidth=2,label='val_losses')
    plt.xlabel('num of example')
    plt.ylabel('loss')
    plt.title('lambda={}'.format(lamda))
    plt.legend(['train', 'cv'])
    plt.show()

if __name__ == '__main__':
    X,y,Xval,yval,Xtest,ytest = load_data()
    #--------Part 1 Learning curves--------
    m = len(X)
    n = len(X[0])
    theta = np.ones((n + 1, 1))
    x = np.append(np.ones((m, 1)), X, axis=1)
    xval=np.append(np.ones((len(Xval),1)),Xval,axis=1)

    k=12
    train_losses=np.zeros(k)
    val_losses=np.zeros(k)
    for i in range(k):
        theta = np.zeros((n + 1, 1))
        ret=my_fmincg(theta,x[0:i+1],y[0:i+1],0)
        #print(ret['fun'])
        theta_pred = ret['x'].reshape(-1, 1)
        train_losses[i]=cost_func(theta_pred,x[0:i+1],y[0:i+1],0)   #随着数据集的增大，训练集误差增加到趋于平缓
        val_losses[i]=cost_func(theta_pred,xval,yval,0)             #验证用全部的数据，验证误差减小到趋于平缓
    plt_learning_curve(train_losses,val_losses)
