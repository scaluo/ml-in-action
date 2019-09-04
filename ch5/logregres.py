from math import exp
import numpy as np
import matplotlib.pyplot as plt

#创建数据集，返回特征数据和标签列
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

#跃阶sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    #数组转换成numpy的mat
    dataMatrix = np.mat(dataMatIn)
    #标签列转换成numpy的mat,并进行转置
    labelMat = np.mat(classLabels).transpose()
    #返回dataMatrix的大小。m为行数,n为列数。
    m, n = np.shape(dataMatrix)
    #移动步长,也就是学习速率,控制更新的幅度。
    alpha = 0.001
    #最大迭代次数
    maxCycles = 500
    #初始化W是N*1，值为1的数组
    weights = np.ones((n,1))
    for k in range(maxCycles):
        #梯度上升矢量化公式
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        #w = w+a*d*e
        weights = weights + alpha * dataMatrix.transpose() * error
    #将矩阵转换为数组，返回权重数组
    return weights.getA()

#随机梯度上升，一次一个样本点
def stocGradAscent0(dataMatIn,classLabels):
    m,n = np.shape(dataMatIn)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatIn[i]*weights))
        error = classLabels[i]-h
        weights = weights+alpha*error*dataMatIn[i]
    return weights

def plotBestFit(wei):
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcode1=[]
    ycode1=[]
    xcode2=[]
    ycode2=[]
    #两种不同分类的数据分别放，画不同颜色的散点
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
    #任意生成X轴上的点，-3到3，步长0.1
    x = np.arange(-3.0,3.0,0.1)
    # w0x0+w1x1+w2x2=0,x0=1  ,   x2=(-w0-w1x1)/w2
    y = (-wei[0]-wei[1]*x)/wei[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataMat,labels = loadDataSet()
    #wei = gradAscent(dataMat,labels)
    wei = stocGradAscent0(np.array(dataMat),labels)
    print(wei)
    plotBestFit(wei)
    #test draw line
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # x=np.array([1,2,3,4,5,6,7,8])
    # y=np.array([48083,41786,36194,35302,33164,39930,47436,43738])
    # ax.plot(x,y)
    # plt.show()