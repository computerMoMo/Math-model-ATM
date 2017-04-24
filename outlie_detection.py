import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import csv
import codecs
import sys
from mpl_toolkits.mplot3d import Axes3D
rng = np.random.RandomState(42)

data_path = 'data/1.csv'
with open(data_path, 'r',encoding='utf-8') as data_reader:
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
print("sample nums:", sample_nums)

x_data = []
for i in range(0, sample_nums):
    sample_data = []
    sample_data.append(nums[i])
    sample_data.append(accs[i])
    sample_data.append(times[i])
    x_data.append(sample_data)
x_data_array = np.asarray(x_data, dtype=np.float32)
print("x data shape:", x_data_array.shape)
acc_time_array = x_data_array[:, 1:].copy()
print("acc time data shape:", acc_time_array.shape)

error_nums = 0
color_data = []

clf = IsolationForest(max_samples=1000, random_state=rng,contamination=0.005).fit(x_data_array)
y_data= clf.predict(x_data_array)
for y_ in y_data:
    if y_ == -1:
        error_nums += 1
        color_data.append('r')
    # elif y_ == 1:
    #     color_data.append('g')
    else:
        color_data.append('b')
print("error nums:", error_nums)

# plt.scatter(x_data_array[:, 1], x_data_array[:, 2], c=color_data)
# plt.title("sample acc-time")
# plt.xlabel('acc')
# plt.ylabel('time')
# plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_data_array[:, 0], x_data_array[:, 1], x_data_array[:, 2], c=color_data)
ax.set_zlabel('time')
ax.set_xlabel('num')
ax.set_ylabel('acc')
plt.show()

