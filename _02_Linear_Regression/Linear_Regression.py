# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
#岭回归函数
def ridge(data):
    # 先求出(A^T*A+cI)xhat
    X, Y = read_data()
    add = np.ones(len(Y)).reshape(len(Y), 1)
    # 将X置为增广矩阵
    X = np.c_[X, add]
    const=0.01
    # 以下为通过最小二乘法得出权重向量w
    temp = np.linalg.inv(np.dot(X.T, X)+const * np.identity(7))
    w = np.dot(temp,np.dot(X.T, Y))
    result = 0.5
    for i in range(len(data)):
        result+=w[i]*data[i]
    result+= w[len(data)]
    return result
#符号函数
def sign(x):
    if x>0:
        return 1
    if x<0:
        return -1
    return 0
#梯度下降法求解lasso回归
def lasso(data):
    X, Y = read_data()
    add = np.ones(len(Y)).reshape(len(Y), 1)
    # 将X置为增广矩阵
    X = np.c_[X, add]
    #设置参数和学习率
    const = 1
    apha=1e-7
    #设置最大学习轮数
    opec=100000
    # 以下为梯度下降法
    w=np.ones(7).reshape((7,1))
    for k in range(opec):
        for i in range(len(w)):
            Yhat = np.dot(w.T, X.T)
            temp=Yhat-Y
            temp2=X[:,i].reshape(len(Y),1)
            w[i] = w[i] - apha * ((1 / len(Y)) * np.dot(temp, temp2) + const * sign(w[i]))

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