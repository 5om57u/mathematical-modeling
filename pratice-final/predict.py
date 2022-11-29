import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体

import matplotlib;

matplotlib.use('TkAgg')
mpl.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 前向计算
def forward_NN(x, w, b):
    h1 = sigmoid(w[0] * x[0] + w[1] * x[1] + b[0])
    h2 = sigmoid(w[2] * x[0] + w[3] * x[1] + b[0])
    h3 = sigmoid(w[4] * x[0] + w[5] * x[1] + b[0])
    print(h1, h2, h3)
    o1 = sigmoid(w[6] * h1 + w[8] * h2 + w[10] * h3 + b[1])
    o2 = sigmoid(w[7] * h1 + w[9] * h2 + w[11] * h3 + b[1])
    return h1, h2, h3, o1, o2


# 反向传递 调整参数
def fit(o1, o2, y, x, w, lrate, epochs):
    for i in range(epochs):
        # 循环迭代 调整参数
        p1 = lrate * (o1 - y[0]) * o1 * (1 - o1)
        p2 = lrate * (o2 - y[1]) * o2 * (1 - o2)
        w[0] = w[0] - (p1 * w[6] + p2 * w[7] * h1 * (1 - h1) * x[0])
        w[1] = w[1] - (p1 * w[6] + p2 * w[7] * h1 * (1 - h1) * x[1])
        w[2] = w[2] - (p1 * w[8] + p2 * w[9] * h2 * (1 - h2) * x[0])
        w[3] = w[3] - (p1 * w[8] + p2 * w[9] * h2 * (1 - h1) * x[1])
        w[4] = w[4] - (p1 * w[10] + p2 * w[11] * h3 * (1 - h3) * x[0])
        w[5] = w[5] - (p1 * w[10] + p2 * w[11] * h3 * (1 - h3) * x[1])
        w[6] = w[6] - p1 * h1
        w[7] = w[7] - p2 * h1
        w[8] = w[8] - p1 * h2
        w[9] = w[9] - p2 * h2
        w[10] = w[10] - p1 * h3
        w[11] = w[11] - p2 * h3
    return w


print("步骤一 初始化参数")

x=[3,6]
y=[0,1]
w=[0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70,0.71,0.72,0.73]
b=[0.3,0.6]
lrate=0.4

print("步骤二 fit")
print("步骤三 预测")
print("真值为", y)
sumDs = []
for epochs in range(0, 101, 5):
    h1, h2, h3, o1, o2 = forward_NN(x, w, b)
    w = fit(o1, o2, y, x, w, lrate, epochs)
    h1, h2, h3, o1, o2 = forward_NN(x, w, b)
    print("迭代", epochs, "次的输出为\n", o1, o2)
    sumDs.append(o1 - y[0] + (o2 - y[1]))

print("画图")
plt.plot(range(0, 101, 5), sumDs)
plt.title("the epoch-error plot for 沈阳温度预测")
plt.xlabel("epochs")
plt.ylabel("totol error")
plt.show()