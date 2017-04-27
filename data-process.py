# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import csv
import os
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 随机数种子
rng = np.random.RandomState(524)

if __name__ == '__main__':
    feb_data_path = 'new-data/'
    nums = []
    accs = []
    times = []
    for file_name in os.listdir(feb_data_path):
        with open(feb_data_path+file_name, 'r') as data_reader:
            csv_reader = csv.reader(data_reader, delimiter='\t')
            head_str = next(csv_reader)
            for row in csv_reader:
                affair_num = float(str(row[-3]).replace(',', ''))
                nums.append(affair_num)

                affair_acc = float(str(row[-2]).replace(',', '')) * 100.0
                accs.append(affair_acc)

                affair_time = float(str(row[-1]).replace(',', ''))
                times.append(affair_time)
    # 业务变化量计算
    num_changes = []
    for i in xrange(0, 10):
        num_changes.append(0.)
    for i in xrange(10, len(nums)):
        num_max = nums[i - 10]
        num_min = nums[i - 10]
        sum = num_max
        for j in xrange(1, 10):
            sum += nums[i - j]
            if num_max < nums[i - j]:
                num_max = nums[i - j]
            if num_min > nums[i - j]:
                num_min = nums[i - j]
        sum -= num_min
        sum -= num_max
        num_changes.append(float(nums[i] - sum / 8.))

    print("nums shape:", len(nums))
    print("num_change:", len(num_changes))

    sample_nums = len(num_changes)
    # 数据预处理归一化
    max_num = max(num_changes)
    min_num = min(num_changes)
    length = max_num - min_num
    for i in xrange(0, sample_nums):
        num_changes[i] = (num_changes[i] - min_num) / length

    max_acc = max(accs)
    min_acc = min(accs)-1
    length = max_acc - min_acc
    for i in xrange(0, sample_nums):
        accs[i] = (accs[i] - min_acc) / length

    max_times = max(times)
    min_times = min(times)-1
    length = max_times - min_times
    for i in xrange(0, sample_nums):
        times[i] = (times[i] - min_times) / length

    # box cox 变化
    accs, lambda_acc = stats.boxcox(np.asarray(accs, dtype=np.float32))
    times, lambda_times = stats.boxcox(np.asarray(times, dtype=np.float32))

    print ("lambda acc:", lambda_acc)
    print ("lambda times:", lambda_times)

    new_train_data = []
    for i in xrange(0, len(nums)):
        sample_data = []
        sample_data.append(num_changes[i])
        sample_data.append(accs[i])
        sample_data.append(times[i])
        new_train_data.append(sample_data)
    new_train_data_array = np.asarray(new_train_data, dtype=np.float32)
    print("new train data shape:", new_train_data_array.shape)
    np.save("test-data-array-box.npy", new_train_data_array)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(new_train_data_array[:, 0], new_train_data_array[:, 1], new_train_data_array[:, 2], c='y')
    ax.set_xlabel('Changes of business volume')
    ax.set_ylabel('Transaction success rate')
    ax.set_zlabel('Transaction response time')

    plt.show()
