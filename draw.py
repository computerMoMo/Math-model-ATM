# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_data_path = 'train-data-array.npy'
    result_data_path = 'result-array.npy'
    train_array = np.load(train_data_path)
    res_array = np.load(result_data_path)

    colors = []
    error_nums = 0
    for res in res_array.tolist():
        if res == -1:
            # 红色代表异常数据
            colors.append('r')
            error_nums += 1
        else:
            # 蓝色代表正常数据
            colors.append('b')
    print ("error sample nums:", error_nums)

    # num-time图
    # plt.scatter(train_array[:, 0], train_array[:, 2], c=colors)
    # plt.title("sample num-time")
    # plt.xlabel('num')
    # plt.ylabel('time')

    # num-acc
    # plt.scatter(train_array[:, 0], train_array[:, 1], c=colors)
    # plt.title("sample num-acc")
    # plt.xlabel('num')
    # plt.ylabel('acc')

    # acc-time
    # plt.scatter(train_array[:, 1], train_array[:, 2], c=colors)
    # plt.title("sample acc-time")
    # plt.xlabel('acc')
    # plt.ylabel('time')

    # 3d图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(train_array[:, 0], train_array[:, 1], train_array[:, 2], c=colors)
    ax.set_zlabel('time')
    ax.set_xlabel('num')
    ax.set_ylabel('acc')

    plt.show()
