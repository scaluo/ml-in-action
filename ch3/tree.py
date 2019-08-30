from math import log
import operator

#创建数据集和特征说明，labels表示特征说明，按照列次序
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt


#划分数据集，axis指定特征列，value指定特征值
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            rec = featVec[:axis]
            rec.extend(featVec[axis+1:])
            retDataSet.append(rec)
    return retDataSet

#选择最佳特征，输入训练数据集
def chooseBestFeatureToSplit(dataSet):
    #计算特征数，减去标签列
    numFeatures = len(dataSet[0])-1      
    #计算数据集的香农熵Ent
    baseEntropy = calcShannonEnt(dataSet)
    #定义最佳信息增益
    bestInfoGain = 0.0
    #定义最佳特征索引
    bestFeature = -1
    #遍历特征列
    for i in range(numFeatures):
        #取特征索引为i的所有数据
        featlist = [item[i] for item in dataSet]
        #取特征索引为i的特征值
        uniqVals = set(featlist)
        #特征索引为I的信息熵 SUM（DV/D)ENt(DV)
        newEntropy = 0.0
        #遍历特征值
        for value in uniqVals:
            #切割数据集（根据特征和特征值）
            subDataSet = splitDataSet(dataSet,i,value)
            #计算DV/D
            prob = len(subDataSet)/float(len(dataSet))
            #SUM（DV/D)ENt(DV)
            newEntropy += prob*calcShannonEnt(subDataSet)
        #特征值对应的信息增益  ENT(D) - SUM（DV/D)ENt(DV)
        infoGain = baseEntropy-newEntropy
        #获取最佳信息增益和最佳特征值
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

#特征值用完，只有标签列，判断是哪个分类，按照分类数量多少判断
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classList.keys:
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

#创建决策树，输入数据集，特征标签数组（对应特征值的定义），递归
def createTree(dataSet,labels):
    #取数据集中标签列
    classList = [data[-1] for data in dataSet]
    #如果标签列中所有标签相同，返回标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #如果特征列都用完了，只有标签列了，就根据标签列的标签多少判断
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #获取最佳特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #获取最佳特征说明
    bestFeatLabel = labels[bestFeat]
    #树顶节点
    myTree = {bestFeatLabel:{}}
    #复制特征说明数组
    subLabels=labels[:]
    #删除最佳特征列
    del(subLabels[bestFeat])
    #最佳特征列的值列表
    featValues = [data[bestFeat] for data in dataSet]
    #最佳特征值
    uniqVals = set(featValues)
    #遍历最佳特征的不同值，递归向下建子树
    for value in uniqVals:
        #subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#第一个参数：决策树数据（DICT），第二个参数：特征标签（数组），第三个参数（输入的测试特征数据）
def classify(inputTree,featLabels,testVec):
    #树的第一个节点
    firstStr = list(inputTree.keys())[0]
    #获取第一个节点下的DICT
    secondDict = inputTree[firstStr]
    #第一个节点的特征索引
    featIndex = featLabels.index(firstStr)
    #遍历第一个节点特征下的KEY
    for key in list(secondDict.keys()):
        #如果测试数据特征输入等于KEY
        if testVec[featIndex] == key:
            #如果节点是DICT，继续递归遍历
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

    # firstStr = list(inputTree.keys())[0]
    # secondDict = inputTree[firstStr]
    # featIndex = featLabels.index(firstStr)
    # for key in secondDict.keys():
    #     if testVec[featIndex] == key:
    #         if type(secondDict[key]).__name__ == 'dict':
    #             classLabel = classify(secondDict[key],featLabels,testVec)
    #         else:
    #             classLabel = secondDict[key]
    # return classLabel


    
if __name__ == "__main__":
    datasets,labels = createDataSet()
    mytree = createTree(datasets,labels)
    print(mytree)