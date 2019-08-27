import numpy as np

def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]          #1 侮辱性 0 非侮辱性
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word:%s is not in vocablist!'%word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                         #计算训练的文档数目
    numWords = len(trainMatrix[0])                          #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)       #文档属于侮辱类的概率
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)  #创建numpy.zeros数组,
    p0Denom = 0.0; p1Denom = 0.0                            #分母初始化为0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                           #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                                               #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom                                  #相除        
    p0Vect = p0Num/p0Denom          
    return p0Vect,p1Vect,pAbusive                           #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) * pClass1             #对应元素相乘
    p0 = sum(vec2Classify * p0Vec) * (1.0 - pClass1)
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    listPosts,listClasses = loadDataSet()
    vocabList = createVocabList(listPosts)
    trainMat = []
    for post in listPosts:
        trainMat.append(setOfWords2Vec(vocabList,post))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(vocabList,testEntry))
    print("classify is:%d"%classifyNB(thisDoc,p0V,p1V,pAb))