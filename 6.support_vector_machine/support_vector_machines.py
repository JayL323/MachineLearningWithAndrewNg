# coding=utf-8
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


def load_data(filename):
    datas = loadmat(filename)
    X = datas['X']
    y = datas['y']

    if np.all(datas.get('Xval')) != None:
        Xval = datas['Xval']
        yval = datas['yval']
        return X, y, Xval, yval
    else:
        return X, y


def plot_data(X, y):
    plt.figure()
    neg_data = X[np.where(y[:, 0] == 0)]
    pos_data = X[np.where(y[:, 0] == 1)]

    plt.scatter(neg_data[:, 0], neg_data[:, 1].tolist(), marker='o', color='red', linewidths=2)
    plt.scatter(pos_data[:, 0], pos_data[:, 1].tolist(), marker='+', color='blue', linewidths=2)
    plt.show()


def plot_pred_data(X, y, model):
    '''画出决策边界'''
    plt.figure()
    neg_data = X[np.where(y[:, 0] == 0)]
    pos_data = X[np.where(y[:, 0] == 1)]

    plt.scatter(neg_data[:, 0], neg_data[:, 1], marker='o', color='red', linewidths=2)
    plt.scatter(pos_data[:, 0], pos_data[:, 1], marker='+', color='blue', linewidths=2)

    xi = [i for i in np.linspace(np.min(X[:, 0] - 0.1), np.max(X[:, 0] + 0.1))]
    xj = [j for j in np.linspace(np.min(X[:, 1] - 0.1), np.max(X[:, 1]) + 0.1)]
    xi, xj = np.meshgrid(xi, xj)
    xi_, xj_ = np.reshape(xi, (-1, 1)), np.reshape(xj, (-1, 1))
    x = np.append(xi_, xj_, axis=1)
    ypred = model.predict(x).reshape(-1, 1)

    plt.contour(xi, xj, ypred.reshape(50, 50), levels=[0])
    plt.show()


def gaussian_kernel(xi, xj, sigma=1):
    all = np.sum((xi - xj) ** 2)
    return np.exp(-all / (2 * sigma * sigma))


def plot_acc(accs):
    plt.figure()
    x = np.linspace(0, len(accs), num=len(accs))
    plt.plot(x, accs, color='red', linewidth=1)
    plt.xlabel('diff params')
    plt.ylabel('acc')
    plt.show()


if __name__ == '__main__':
    # --------Part 1 Example Dataset 1--------
    # filename='data/ex6data1.mat'
    # X,y=load_data(filename)
    # #plot_data(X,y)
    # #kernel=高斯核函数（RBF）,多项式核函数'ploy',线性核函数：linear
    # #核的作用类似于sigmoid在逻辑分类中的作用
    # model=SVC(C=1,kernel='linear',degree=3)              #设置模型参数
    # model.fit(X,y)                                    #模型训练

    # plot_pred_data(X,y,model)

    # --------Part 2 SVM with Gaussian Kernels--------
    # --------Part 2.1 Implement Gaussian Kernels--------
    # print(X[0],X[1])
    # xi = np.array([1, 2, 1])
    # xj = np.array([0, 4, -1])
    # sigma = 2
    # ret=gaussian_kernel(xi,xj,sigma=2)

    # --------Part 2.2 Example Dataset 2--------
    # X_d2,y_d2=load_data('data/ex6data2.mat')
    # #plot_data(X_d2,y_d2)
    # model_rbf=SVC(C=50,kernel='rbf',degree=3,gamma=0.1)   #C越大，对异常点越敏感;gamma越小，对异常点越敏感
    # model_rbf.fit(X_d2,y_d2.ravel())
    # plot_pred_data(X_d2,y_d2,model_rbf)

    # --------Part 2.3 Example Dataset 3--------
    # C选择（C非常大，低偏差，高方差；C非常小，高偏差，低方差）
    # σ选择（对于高斯核函数，σ大，高偏差，低方差，反之亦然）
    X_d3, y_d3, Xval_d3, yval_d3 = load_data('data/ex6data3.mat')
    prams = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    accs = []
    for C in prams:
        for gamma in prams:
            model_rbf = SVC(C=C, kernel='rbf', degree=3, gamma=gamma)
            model_rbf.fit(X_d3, y_d3.ravel())
            # yval_pred=model_rbf.predict(Xval_d3).reshape(-1,1)
            # acc=np.sum(yval_pred==yval_d3,dtype=np.int)
            # accs.append(acc)
    # index=accs.index(max(accs))
    # print('C={},gamma={}'.format(prams[int(index/len(prams))],prams[index%len(prams)]))
    # plot_acc(accs)
    # 最合适的参数：C=3,gamma=30
    model_rbf = SVC(C=3, kernel='rbf', degree=3, gamma=30)
    model_rbf.fit(Xval_d3, yval_d3.ravel())
    plot_pred_data(Xval_d3, yval_d3, model_rbf)
