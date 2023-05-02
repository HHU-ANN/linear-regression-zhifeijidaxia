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
    add = np.ones(len(Y)).reshape(len(Y), 1)
    # 将X置为增广矩阵
    X = np.c_[X, add]
    const=0.01
    # 以下为通过最小二乘法得出权重向量w
    temp = np.linalg.inv(np.matmul(X.T, X)+ const * np.identity(7))
    w = np.matmul(np.matmul(temp, X.T), Y)

    result = 0
    for i in range(len(data)):
        result+=w[i]*data[i]
    result+= w[len(data)]
    return result

def lasso(data):
    # 先求出(A^T*A+cI)xhat
    X, Y = read_data()
    add = np.ones(len(Y)).reshape(len(Y), 1)
    # 将X置为增广矩阵
    X = np.c_[X, add]
    const = 0.01
    # 以下为通过最小二乘法得出权重向量w
    temp = np.linalg.inv(np.matmul(X.T, X) + const * np.identity(7))
    w = np.matmul(np.matmul(temp, X.T), Y)

    result = 0
    for i in range(len(data)):
        result += w[i] * data[i]
    result += w[len(data)]
    return result


# 原路径为def read_data(path='./data/exp02/'):
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

