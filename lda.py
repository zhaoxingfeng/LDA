# coding:utf-8
"""
作者：zhaoxingfeng	日期：2017.06.08
功能：线性判别分析，Linear Discriminant Analysis（LDA），二类分类和降维
版本：2.0
参考文献：
[1] LeftNotEasy-Wangda Tan.机器学习中的数学(4)-线性判别分析（LDA）, 主成分分析(PCA)[DB/OL].
    http://www.cnblogs.com/LeftNotEasy/archive/2011/01/08/lda-and-pca-machine-learning.html,2011-01-08.
[2] JerryLead.线性判别分析（Linear Discriminant Analysis）（一）[DB/OL].
    http://www.cnblogs.com/jerrylead/archive/2011/04/21/2024384.html,2011-04-21.
"""
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compiler.ast import flatten
import random

# 类内距离
def within_class_SW(data, mean_class):
    n, m = data.shape
    SW = np.mat(np.zeros((m-1, m-1)))
    for row in data.values:
        diff = row[:-1] - mean_class.ix[row[-1]].values
        diff = np.mat(diff)
        SW += diff.T * diff
    return SW

# 类间距离
def between_class_SB(data, mean_class):
    n, m = data.shape
    # 所有类别样本均值
    mean_all = data.mean().values[:-1]
    # 每一类别样本数量
    count_class = data.groupby([m-1]).count()
    # 类别
    clss = data[m-1].drop_duplicates().values
    SB = np.mat(np.zeros((m-1, m-1)))
    for cls in clss:
        diff = mean_class.ix[cls].values - mean_all
        diff = np.mat(diff)
        SB += (diff.T * diff) * count_class.ix[cls][0]
    return SB

def mylda(data, n_components=2):
    n, m = data.shape
    clss = data[m - 1].drop_duplicates().values
    if n_components > len(clss)-1:
        print "dim is big!"
        return None
    # 每个类别下特征的均值
    mean_class = data.groupby([m-1]).mean()
    sw = within_class_SW(data, mean_class)
    sb = between_class_SB(data, mean_class)
    s = np.linalg.inv(sw) * sb
    eigVal, eigVect = np.linalg.eig(np.mat(s))
    # 挑选若干特征
    eig_pairs = [[eigVal[i].real, eigVect[:, i].real] for i in range(len(eigVal))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    w = np.zeros((m-1, 1))
    for x in eig_pairs[:n_components]:
        w = np.hstack((w, x[1]))
    w = w[:, 1:n_components+1]
    returnMat = np.mat(data)[:, :-1] * w
    return w, returnMat

# 输入测试样本，分类到class1和class2
def classify(query_data, data, newMat, w):
    n, m = data.shape
    axis_x = flatten(data[0].tolist())
    axis_y = flatten(data[1].tolist())
    label = data.values[:, -1]
    color = ['b', 'r', 'g', 'y', 'c']
    # 原始二维散点图
    plt.subplot(2, 1, 1)
    for i in range(n):
        plt.scatter(axis_x[i], axis_y[i], c=color[int(label[i])], marker='o', s=5)
    mean_class = data.groupby([m - 1]).mean().values
    count_class = data.groupby([m - 1]).count().values[:, 0]
    w0 = sum([x*y*w for x, y in zip(mean_class, count_class)]) / n
    for dt in query_data:
        if dt * w - w0[0, 0] > 0:
            print "class1"
        else:
            print "class2"
        plt.title('raw data')
    # 变换后只有一维，为了显示直观，y轴随机取值0-1
    plt.subplot(2, 1, 2)
    axis_x = flatten(newMat[:, 0].tolist())
    axis_y = [random.random() for _ in range(n)]
    for j in range(n):
        plt.scatter(axis_x[j], axis_y[j], c=color[int(label[j])], marker='o', s=5)
    plt.plot([w0[0, 0]] * n, axis_y)
    plt.title('new data')
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv(r'iris12.txt', header=None)
    w, newMat = mylda(df, 1)
    classify([[5.1, 3.8, 1.9, 0.4], [6.4, 3.2, 4.5, 1.5]], df, newMat, w)
