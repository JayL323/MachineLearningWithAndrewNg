# coding=utf-8

from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2


def plot_data(X, marker='o', color='blue'):
    plt.scatter(X[:, 0], X[:, 1], marker=marker, color=color)
    # plt.show()


def feature_normalize(X):
    x = X.copy().astype(np.float)  # 标准化时要注意数据的类型，一定是float
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    for i in range(len(mean)):
        x[:, i] = (x[:, i] - mean[i]) / std[i]
    return mean, std, x


def draw_line(p1, p2, plot_kwargs=""):
    x = np.array([p1[0], p2[0]])
    y = np.array([p1[1], p2[1]])
    plt.plot(x, y, plot_kwargs)
    # plt.show()


def project_data(x_norm, U, K):
    # 将数据根据U,K降低到低维空间
    u_reduce = U[:, 0:K]
    return np.dot(x_norm, u_reduce)


def recover_data(pro_data, U, K):
    # 将低维空间恢复到原来的高维数据空间
    u_reduce = U[:, 0:K]
    return np.dot(pro_data, u_reduce.T)


def gray(f):
    f_new = (f - np.min(f)) / (np.max(f) - np.min(f)) * 255
    return f_new


def show_top_k_faces(X, k=100, width=32, height=32):
    # width, height = 10, 10
    lines, rows = int(np.sqrt(k)), int(np.sqrt(k))
    img = np.zeros((rows * width, lines * height))
    # max_val=np.max(np.abs(x))
    for line in range(lines):
        for row in range(rows):
            x = X[line * rows + row].reshape(width, height, order='F')
            img[line * height:(line + 1) * height, row * width:(row + 1) * width] = gray(x)  # /max_val
    cv2.imwrite('data/top-100_faces.png', img)


if __name__ == '__main__':
    # ------Part 1 Example Dataset------
    datas = loadmat('data/ex7data1.mat')
    X = datas['X']
    m = len(X)
    # plot_data(X)
    # plt.show()

    # ------Part 2 Implementing PCA------
    # 不是所有的情况都适合PCA,具体要看数据的分布
    mean, std, x_norm = feature_normalize(X)
    # (1)compute the covariance matrix of the data, which is given by
    SIGMA = np.dot(x_norm.T, x_norm) / m  # 2*2
    # (2)run SVD on it to compute the principal components.
    U, S, V = np.linalg.svd(SIGMA)  # U是主成分；S是对角阵
    # print(U,S)
    # draw_line(mean,mean+std*np.dot(S[0],U[:,0].T),plot_kwargs='-k')        #画出新的坐标轴
    # draw_line(mean,mean+std*np.dot(S[1],U[:,1].T),plot_kwargs='-k')
    # plt.show()

    # ------Part 3 Dimensionality Reduction with PCA------
    K = 1
    pro_data = project_data(x_norm, U, K)
    # print(pro_data[0])
    rec_data = recover_data(pro_data, U, K)
    # print(rec_data[0])
    plt.figure()
    plot_data(x_norm)
    rec_data_init = mean + std * rec_data
    # plot_data(rec_data,marker='+',color='red')
    # #为什么我画出的新的x,y并不是垂直
    # for i in range(m):
    #     draw_line(x_norm[i],rec_data[i],'--k')   #画虚线
    # plt.show()

    # ------Part 4 Face Image Dataset------
    faces_datas = loadmat('data/ex7faces.mat')
    X = faces_datas['X']  # 5000*1024(=32*32)
    m = len(X)
    # visual first 100 faces
    mean, std, x_norm = feature_normalize(X)
    # run pca
    SIGMA = np.dot(x_norm.T, x_norm) / m
    U, S, V = np.linalg.svd(SIGMA)
    k = 100  # 取前k个主成分
    # show_top_k_faces(np.dot(np.dot(x_norm,U[:,0:k]),U[:,0:k].T), k=k)
    # 训练的时候可以只用前k维训练
    show_top_k_faces(np.dot(x_norm, U[:, 0:k]), k=k, width=10, height=10)

    # ------Part 5 PCA for visualization(只完成了2维可视化)------
    png = cv2.imread('data/bird_small.png').reshape(-1, 3, order='F')
    m = len(png)
    mean, std, x_norm = feature_normalize(png)
    SIGMA = np.dot(x_norm.T, x_norm) / m
    U, S, V = np.linalg.svd(SIGMA)

    figure = plt.figure()
    top_k = 2
    x_norm_ruduce = project_data(x_norm, U, top_k)
    plt.scatter(x_norm_ruduce[0:100, 0], x_norm_ruduce[0:100, 1], color='red', marker='+')
    plt.show()
