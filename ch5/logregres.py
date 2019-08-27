from math import exp
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                       #创建标签列表
    fr = open('testSet.txt')                                            #打开文件   
    for line in fr.readlines():                                         #逐行读取
        lineArr = line.strip().split()                                  #去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])     #添加数据
        labelMat.append(int(lineArr[2]))                                #添加标签
    fr.close()                                                          #关闭文件
    return dataMat, labelMat                                            #返回

"""
函数说明:sigmoid函数

Parameters:
    inX - 数据
Returns:
    sigmoid函数
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
"""
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

"""
函数说明:梯度上升算法

Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                      #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                          #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                         #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                       #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                     #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                               #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                                               #将矩阵转换为数组，返回权重数组

def plotBestFit(wei):
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcode1=[]
    ycode1=[]
    xcode2=[]
    ycode2=[]
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcode1.append(dataArr[i,1])
            ycode1.append(dataArr[i,2])
        else:
            xcode2.append(dataArr[i,1])
            ycode2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xcode1,ycode1,s=30,c='red',marker='s')
    ax.scatter(xcode2,ycode2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-wei[0]-wei[1]*x)/wei[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

