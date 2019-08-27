import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#KNN计算
def classify0(inX,dataSet,labels,k):
    dataSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**5
    sortedDistIndicis = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicis[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]

#从文件中读取数据
def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLabelVector

#归一化数据处理
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#测试错误率
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("The classifier came back with :%s,the real answer is:%s"%(classifierResult,datingLabels[i]))
        if (classifierResult!=datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is:%f" % (errorCount/float(numTestVecs)))

if __name__== '__main__':
    # datasets,labels = createDataSet()
    # print(classify0([1.0,1.2],datasets,labels,3)) 
    #transMat,transLabs = file2matrix('datingTestSet2.txt')
    datingClassTest()

