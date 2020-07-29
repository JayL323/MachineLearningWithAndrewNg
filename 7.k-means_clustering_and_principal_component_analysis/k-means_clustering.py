# coding=utf-8

'''
    k-means 3步：
    （1）初始化中心点
    （2）对于每个训练数据，找到距离最近的点，返回对应的类别
    （3）对于每个类别，计算新的中心点
'''
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2


# 初始化中心点
def k_means_init_centroids(X, K):
    m = X.shape[0]
    randidx = np.random.permutation(m)  # 将index随机打乱
    centroids = X[randidx[0:K], :]  # 取得随机打乱后的前K个X    return centroids
    return centroids


# 对于每个点，找到最近的中线点作为一个聚类
def find_closest_centroids(X, centroids):
    idx = np.zeros(len(X), dtype=np.int)
    for i, x in enumerate(X):
        dist = np.sum((x - centroids) ** 2, axis=1)
        dist = dist.tolist()
        idx[i] = dist.index(min(dist))
    return idx


# 对于每个聚类，找到新的中心点
def computeMeans(X, idx, K):
    means = np.zeros((K, X.shape[1]))
    for i in range(K):
        indexs = np.where(idx == i)
        x = X[indexs]
        if len(x) != 0:
            means[i] = np.mean(x, dtype=np.int, axis=0)
    return means


def run_kmeans(epoches, X, centroids, K):
    centroids_all = [centroids]
    for i in range(epoches):
        idx = find_closest_centroids(X, centroids)
        means = computeMeans(X, idx, K)
        centroids = means
        centroids_all.append(centroids)
    return centroids_all


def plot_data(X, centroids, K):
    plt.figure()
    centroids = np.array(centroids).reshape(-1, K * X.shape[1])
    centroid = centroids[-1].reshape(-1, X.shape[1])
    print(centroid)
    idx = find_closest_centroids(X, centroid)
    markers = ['x', 'o', '+']
    colors = ['red', 'blue', 'green']
    for i in range(3):
        class_i = np.where(idx == i)
        class_i_data = X[class_i]
        plt.scatter(class_i_data[:, 0], class_i_data[:, 1], marker=markers[i], color=colors[i])
        plt.plot(centroids[:, i * 2], centroids[:, i * 2 + 1], marker='X', color='black')
    plt.title('Iteration number 10')
    plt.show()


if __name__ == '__main__':
    # --------Part 1 Implementing K-means--------
    # k-means算法的最终的中心点与初始值由很大关系；一般选择不同的初始值来确定最好的聚类中心
    K = 2
    X = np.arange(1, 11).reshape(10, 1)
    centroids = np.arange(1, 3).reshape(2, 1)
    idx = find_closest_centroids(X, centroids)
    print(idx)
    centroids_all = [centroids]
    for i in range(20):
        idx = find_closest_centroids(X, centroids)
        means = computeMeans(X, idx, K)
        # print(means)
        centroids = means

    # --------Part 2 K-means on example dataset--------
    datas = loadmat('data/ex7data2.mat')
    X = datas['X']  # 50*2
    K = 3
    # centroids=np.zeros((K,X.shape[1]))
    centroids = np.random.rand(K, X.shape[1])
    # centroids_all=run_kmeans(10,X,centroids,K)
    # #print(centroids_all)
    # plot_data(X,centroids_all,K)

    # --------Part 3 Random initialization--------
    centroids = k_means_init_centroids(X, K)
    # centroids_all = run_kmeans(10, X, centroids, K)
    # print(centroids_all)
    # plot_data(X, centroids_all, K)

    # --------Part 4 Image compression with K-means--------
    # png=cv2.imread('data/bird_small.png')
    # K=16
    # png=png.reshape(-1,3)
    # centroids = k_means_init_centroids(png, K)
    # centroids_all = run_kmeans(10, png, centroids, K)
    # #png_data=loadmat('data/bird_small.mat')
    # centroid=np.array(centroids_all[-1]).reshape(-1,3)
    # idx=find_closest_centroids(png,centroid)
    # for i in range(K):
    #     png[np.where(idx==i)]=centroid[i]
    # png=png.reshape(128,128,3)
    # cv2.imwrite('data/bird_small_comp.png',png)

    # --------Part 5  Use your own image--------
    # 需要裁剪自己的图像；自己选择合适的K
    K = 3
    img = cv2.imread('data/618629e1a5b302aca78d0de6447180a3.jpg')
    print(img.shape)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    print(img.shape)
    img = img.reshape(-1, 3)
    centroids = k_means_init_centroids(img, K)
    centroids_all = run_kmeans(10, img, centroids, K)
    centroid = np.array(centroids_all[-1]).reshape(-1, 3)
    idx = find_closest_centroids(img, centroid)
    for i in range(K):
        img[np.where(idx == i)] = centroid[i]
    img = img.reshape(128, 128, 3)
    cv2.imwrite('data/618629e1a5b302aca78d0de6447180a3_comp.jpg', img)
