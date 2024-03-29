import numpy as np
import random

#创建训练数据集和标签数组
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    #1 侮辱性 0 非侮辱性
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

#创建单词表，所有不重复单词的单词表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#在单词表中标记是否出现的标记，1为出现，0为没出现，参数一是单词表，参数二为输入文章
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word:%s is not in vocablist!'%word)
    return returnVec

#文档词袋模型
def bagOfWords2VecMN(vocaList,imputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word:%s is not in vocablist!'%word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    
    #计算训练的文档数目
    numTrainDocs = len(trainMatrix)
    #计算每篇文档的词条数，这个传入的是已经标记过单词表的，每条记录的长度都是单词表的长度
    numWords = len(trainMatrix[0])
    #文档属于侮辱类的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #创建numpy.zeros数组,
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    #所有非侮辱性的词汇出现总和
    p0Denom = 2.0
    #所有侮辱性的词汇出现总和
    p1Denom = 2.0                            #分母初始化为0.0
    for i in range(numTrainDocs):
        #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #相除
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
    return p0Vect,p1Vect,pAbusive

#分类，参数一为已向量化的文档，参数二，侮辱性的概率数组，参数三，非侮辱性数组，参数四，侮辱性的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) +np.log(pClass1)             #对应元素相乘
    p0 = sum(vec2Classify * p0Vec) +np.log(1.0 - pClass1)
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

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

#邮件分类,spam,ham
def spamTest():
    docList = []
    classList = []
    fullText = []
    #遍历两个目录，有侮辱性的和无侮辱性的，创建文档LIST和标签LIST,fullText没用到,原TXT文件是ANSI的，要用8859编码打开
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt'%i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'%i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #简历词汇表
    vocabList = createVocabList(docList)
    
    
    #用10条记录做测试集,书上的没运行通过，改成另一种方式
    trainingSet = list(range(len(docList)))
    testSet = []
    iset = random.sample(trainingSet,10)
    for i in iset:
        testSet.append(i)
        trainingSet.remove(i)
    trainMat = []
    trainClasses = []

    #训练
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))

    #测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount+=1
    print("The error rate is :%f"%(float(errorCount)/len(testSet)))


if __name__=='__main__':
    import chardet
    #testingNB()
    spamTest()
    