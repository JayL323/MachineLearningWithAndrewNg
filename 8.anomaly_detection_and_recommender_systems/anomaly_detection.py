#coding=utf-8

'''异常检测：用的算法是多元高斯函数；需要少量标签来选择何使阈值'''

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    datas = loadmat('data/ex8data1.mat')
    X=datas['X']
    Xval=datas['Xval']
    yval=datas['yval']
    return X,Xval,yval

def estimate_gaussian(X):
    '''计算平均值和方差'''
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0,ddof=1)
    #simga=std;sigma2=std**2
    #sigma2=np.sum((X-mean)**2,axis=0)/(len(X)-1)
    return mean,std

#高斯分布；正太分布
def gaussian_func(X,mean,std):
    #计算每类特征mean,std,多元高斯分布
    '''多元egaussian (X, mu, Sigma2)计算算例X在参数为mu和Sigma2的多元高斯分布下的概率密度
    函数。如果Sigma2是一个矩阵，则将其视为协方差矩阵。如果Sigma2是一个向量，它被视为每个维度的
    方差的\sigma^2值(对角协方差矩阵)'''
    coefficient=1/(np.sqrt(2*np.pi*(std**2)))
    p=np.zeros(X.shape)
    for i in range(len(X)):
        p[i]=coefficient*np.exp(-(X[i]-mean)**2/(2*std**2))
    p_mul=np.ones(len(p))
    for i in range(len(p)):
        for j in p[i]:
            p_mul[i]=p_mul[i]*j
    return p_mul
def multivariate_gaussian(x, mu, sigma2):
    '''多元高斯分布'''
    k = mu.size
    if len(sigma2.shape) == 1:
        sigma2 = np.diag(sigma2)
    err = x - mu
    p = (1 / (np.power(2 * np.pi, k / 2) * np.power(np.linalg.det(sigma2), 1 / 2))) \
        * np.exp((-1 / 2) * np.sum(np.dot(err, np.linalg.pinv(sigma2)) * err, axis=1))
    return p

def select_threshold(pval,yval):
    '''to select the threshold " using the F1 score on a cross validation set.'''
    #虽然结果是一样的，但是不同的epsilon，会产生不同的异常值
    best_f1=0
    best_epsilon=0
    _f=0

    pval=np.ravel(pval)
    yval=np.ravel(yval)
    step_size = (np.max(pval) - np.min(pval)) / 1000

    for epsilon in np.arange(np.min(pval),np.max(pval),step_size):
        pred_val=pval<epsilon
        tp=np.sum(np.logical_and(pred_val==1,yval==1))
        fp=np.sum(np.logical_and(pred_val==1,yval==0))
        fn=np.sum(np.logical_and(pred_val==0,yval==1))
        if tp:
            #有预测正确的
            presion=tp/(tp+fp)
            recall=tp/(tp+fn)
            _f1=2*presion*recall/(presion+recall)
        else:
            #没有预测正确的
            _f1=0
        if _f1>best_f1:
            best_f1=_f1
            best_epsilon=epsilon

    return best_epsilon,best_f1

if __name__ == '__main__':
    #数据加载
    X, Xval, yval=load_data()
    #plt.figure()
    #plt.scatter(X[:,0],X[:,1],marker='o',color='blue')
    #plt.show()

    #--------Part 1 Gaussian distribution--------
    mean,std=estimate_gaussian(X)
    p=gaussian_func(X,mean,std)

    #--------Part 2 Estimating parameters for a Gaussian--------
    #https://www.cnblogs.com/bingjianing/p/9117330.html（多元高斯分布）
    # plt.figure()
    # plt.scatter(X[:,0],X[:,1],marker='o',color='blue')
    # x_t=np.arange(0,30,0.5)
    # y_t=np.arange(0,30,0.5)
    # x_plot,y_plot=np.meshgrid(x_t,y_t)
    # #x_temp = np.vstack((np.ravel(x_plot), np.ravel(y_plot))).T
    # a = np.append(x_plot.reshape(-1,1), y_plot.reshape(-1,1), axis=1)
    # z = gaussian_func(a,mean,std).reshape(x_plot.shape)
    # #z_t=np.reshape(multivariate_gaussian(a,mean,std**2),x_plot.shape)
    # plt.contour(x_plot,y_plot,z,np.logspace(-20, -2, 7))
    # plt.show()

    #--------Part 3 Selecting the threshold,--------
    #select a threshold based on a cross validation set
    #using the F1 score on a cross validation set
    # pval=gaussian_func(Xval,mean,std)
    # best_epsilon,best_f1=select_threshold(pval,yval)
    # print(best_epsilon,best_f1)
    # plt.figure()
    # _pval=gaussian_func(X,mean,std)
    # plt.scatter(X[:, 0], X[:, 1], marker='o', color='blue')
    # anomaly_point=X[np.where(_pval<best_epsilon)]
    # # 把 corlor 设置为空，通过edgecolors来控制颜色
    # plt.scatter(anomaly_point[:,0],anomaly_point[:,1], color='', marker='o', edgecolors='r', s=200)
    # x_t = np.arange(0, 30, 0.5)
    # y_t = np.arange(0, 30, 0.5)
    # x_plot, y_plot = np.meshgrid(x_t, y_t)
    # a = np.append(x_plot.reshape(-1, 1), y_plot.reshape(-1, 1), axis=1)
    # z = gaussian_func(a, mean, std).reshape(x_plot.shape)
    # plt.contour(x_plot, y_plot, z, np.logspace(-20, -2, 7))
    # plt.show()

    #--------Part 4 High dimensional dataset--------
    datas=loadmat('data/ex8data2.mat')
    X,Xval,yval=datas['X'],datas['Xval'],datas['yval']
    mean,std=estimate_gaussian(X)
    pval=gaussian_func(Xval,mean,std)
    best_epsilon,best_f1=select_threshold(pval,yval)
    print(best_epsilon,best_f1)
