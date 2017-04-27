# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import csv
import os
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

# 随机数种子
rng = np.random.RandomState(524)

if __name__ == '__main__':
    # 异常点数据的比例
    error_ratio = 0.005

    # 读取数据
    # data_path = 'data/'
    # nums = []
    # accs = []
    # times = []
    # for file_name in os.listdir(data_path):
    #     with open(data_path+file_name) as data_reader:
    #         csv_data_reader = csv.reader(data_reader, delimiter=',')
    #         csv_head = next(csv_data_reader)
    #         for row in csv_data_reader:
    #             affair_num = float(row[-3])
    #             nums.append(affair_num)
    #
    #             affair_acc = float(row[-2].strip('%'))
    #             accs.append(affair_acc)
    #
    #             affair_time = float(row[-1])
    #             times.append(affair_time)
    # print ("sample nums:", len(nums))
    #
    # # 生成训练数据
    # x_data = []
    # for i in xrange(0, len(nums)):
    #     sample_data = []
    #     sample_data.append(nums[i])
    #     sample_data.append(accs[i])
    #     sample_data.append(times[i])
    #     x_data.append(sample_data)
    # x_data_array = np.asarray(x_data, dtype=np.float32)
    # print("train data shape:", x_data_array.shape)

    x_data_array = np.load('feb-train-data-array.npy')
    x_test_array = np.load('test-data-array-box.npy')
    print("train data shape:", x_data_array.shape)
    print("test data shape:", x_test_array.shape)

    # 使用isolation forest 预测
    iso_forest = IsolationForest(max_samples=256, random_state=rng, contamination=error_ratio)
    iso_forest.fit(x_data_array)
    y_iso = iso_forest.predict(x_test_array)

    # 使用elliptic envelope 预测
    ell_envelope = EllipticEnvelope(contamination=error_ratio, random_state=rng)
    ell_envelope.fit(x_data_array)
    y_ell = ell_envelope.predict(x_test_array)

    # 进行组合预测
    error_nums = 0
    pre_results = []
    for i in xrange(0, x_test_array.shape[0]):
        if y_iso[i] == -1 and y_ell[i] == -1:
            error_nums += 1
            pre_results.append(-1)
        else:
            pre_results.append(1)
    print ("error sample nums:", error_nums)

    # 存储结果
    print ("save results")
    pre_results_array = np.asarray(pre_results, dtype=np.int8)
    np.save("result-array.npy", pre_results_array)
    print ("result array shape:", pre_results_array.shape)
    # np.save("train-data-array.npy", x_data_array)
    # print ("train data array shape:", x_data_array.shape)
