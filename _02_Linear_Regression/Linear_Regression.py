# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    # 先求出(A^T*A+cI)xhat
    X, Y = read_data()
    X = X.transpose()
    add = np.ones(len(Y)).reshape(1, len(Y))
    # 将X置为增广矩阵
    X = np.r_[X, add]
    const = 0.6
    # 以下为通过最小二乘法得出权重向量w
    temp = np.matmul(X, X.transpose()) + const * np.identity(7)
    temp2 = np.linalg.pinv(temp)
    w = np.matmul(np.matmul(temp2, X), Y)

    # 将data函数增添一列1扩为增广矩阵
    add2 = np.ones(len(data)).reshape(len(data), 1)
    data = np.c_[data, add2]
    # result数组
    result = np.zeros(len(data))
    for i in range(len(data)):
        result[i] = np.dot(w, data[i])
    return result


def lasso(data):
    return data


# 原路径为def read_data(path='./data/exp02/'):
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
