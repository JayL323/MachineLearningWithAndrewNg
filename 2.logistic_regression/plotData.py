# coding=utf-8
'''Function to plot 2D classification data'''

import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, y):
    pos = np.where(y[:, 0] == 1)
    neg = np.where(y[:, 0] == 0)
    plt.figure(figsize=(6, 6))
    pos_data = X[pos]
    x_pos_data = pos_data[:, 0]
    y_pos_data = pos_data[:, 1]
    plt.scatter(x_pos_data, y_pos_data, marker='x', linewidths=2, color='red')

    neg_data = X[neg]
    x_neg_data = neg_data[:, 0]
    y_neg_data = neg_data[:, 1]
    plt.scatter(x_neg_data, y_neg_data, marker='*', linewidths=2, color='blue')
    plt.xlabel('Exam1 Score')
    plt.ylabel('Exam2 Score')
    plt.show()
