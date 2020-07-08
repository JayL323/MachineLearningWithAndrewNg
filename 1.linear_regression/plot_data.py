# coding=utf-8
import matplotlib.pyplot as plt


def plot_data(x, y):
    plt.figure(figsize=(6, 6))
    #              记号形状       颜色           点的大小    设置标签
    plt.scatter(x, y, marker='x', color='red', s=40, label='f')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.legend(loc='best')
    plt.show()


def plot_loss(x, y):
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color='red', linewidth='2')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def plot_data(x, y1, y2):
    plt.figure(figsize=(6, 6))
    #              记号形状       颜色           点的大小    设置标签
    plt.scatter(x, y1, marker='x', color='red', s=40, label='f')
    plt.plot(x, y2, color='blue', linewidth='2')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.legend(loc='best')
    plt.show()

