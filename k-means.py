# -*- coding:utf-8 -*-
from __future__ import print_function
import csv
import numpy as np
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

if __name__ == '__main__':
    data_path = 'data/1.csv'
    # 读取数据
    with open(data_path, 'r') as data_reader:
        data_csv = csv.reader(data_reader, delimiter=',')
        data_head = next(data_csv)
        nums = []
        accs = []
        times = []
        for row in data_csv:
            affair_nums = float(row[-3])
            if affair_nums == 0.:
                affair_nums += 1.
            nums.append(affair_nums)

            affair_acc = float(row[-2].strip('%'))
            if affair_acc == 0.:
                affair_acc += 1.
            accs.append(affair_acc)

            affair_time = float(row[-1])
            if affair_time == 0.:
                affair_time += 1.
            times.append(affair_time)
    sample_nums = len(nums)
    print ("sample nums:", sample_nums)

    # # box cox 正态变换
    nums, _ = stats.boxcox(np.asarray(nums, dtype=np.float32))
    # accs, _ = stats.boxcox(np.asarray(accs, dtype=np.float32))
    # times, _ = stats.boxcox(np.asarray(times, dtype=np.float32))

    # 数据预处理归一化
    max_num = max(nums)
    min_num = min(nums)
    length = max_num - min_num
    for i in xrange(0, sample_nums):
        nums[i] = (nums[i]-min_num)/length

    max_acc = max(accs)
    min_acc = min(accs)
    length = max_acc-min_acc
    for i in xrange(0, sample_nums):
        accs[i] = (accs[i]-min_acc)/length

    max_times = max(times)
    min_times = min(times)
    length = max_times-min_times
    for i in xrange(0, sample_nums):
        times[i] = (times[i]-min_times)/length

    x_data = []
    for i in xrange(0, sample_nums):
        sample_data = []
        sample_data.append(nums[i])
        sample_data.append(accs[i])
        sample_data.append(times[i])
        x_data.append(sample_data)

    # 进行聚类
    x_data_array = np.asarray(x_data, dtype=np.float32)
    print("x data shape:", x_data_array.shape)
    acc_time_array = x_data_array[:, 1:].copy()
    print("acc time data shape:", acc_time_array.shape)

    y_data = KMeans(n_clusters=2).fit_predict(x_data_array)
    error_nums = 0
    color_data = []
    for y_ in y_data:
        if y_ == 0:
            error_nums += 1
            color_data.append('r')
        # elif y_ == 1:
        #     color_data.append('g')
        else:
            color_data.append('b')
    print ("error nums:", error_nums)

    # plt.scatter(x_data_array[:, 0], x_data_array[:, 2], c=color_data)
    # plt.title("sample num-time")
    # plt.xlabel('num')
    # plt.ylabel('time')

    # plt.scatter(x_data_array[:, 0], x_data_array[:, 1], c=color_data)
    # plt.title("sample num-acc")
    # plt.xlabel('num')
    # plt.ylabel('acc')

    plt.scatter(x_data_array[:, 1], x_data_array[:, 2], c=color_data)
    plt.title("sample acc-time")
    plt.xlabel('acc')
    plt.ylabel('time')

    # plt.scatter(acc_time_array[:, 0], acc_time_array[:, 1], c=color_data)
    # plt.title("acc-time")
    # plt.xlabel('acc')
    # plt.ylabel('time')

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x_data_array[:, 0], x_data_array[:, 1], x_data_array[:, 2], c=color_data)
    # ax.set_zlabel('time')
    # ax.set_xlabel('num')
    # ax.set_ylabel('acc')

    plt.show()
