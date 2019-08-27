import numpy as np
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename) as fp:
        for line in fp.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]),float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj = H
    if L>aj:
        aj = L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m,n = np.shape(dataMat)
    alphas = np.mat(np.zeros(m,1))
    iters = 0
    while (iters<maxIter):
        alphaPairChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas,labelMat).T*(dataMat*dataMat[i,:].T))+b
            Ei = fxi - float(labelMat[i])
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or 
                ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                j=selectJrand(i,m)
                