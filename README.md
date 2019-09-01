# 《机器学习实战》练习代码
书中代码是python2+的，练习代码都用python3+实现  
## ch2：KNN算法  
计算特征数据间的距离公式  
![image](http://latex.codecogs.com/gif.latex?d=\sqrt{{(x-x1)}^2+{(x-x2)}^2...{(x-xi)}^2})

## ch3：决策树  
决策树最重要的两个公式为信息熵和信息增益的计算  
设样本集合D中的第K类样本所占的比例是Pk,则信息熵的计算公式：  
![image](http://latex.codecogs.com/gif.latex?Ent(D)=-\sum_{k=1}^{|y|}P_{k}log_{2}P_{k})  
假设离散属性（特征）a有V个可能的取值，若使用a来对样本数据基D进行划分，则就有V个分支节点，其中第v个取值包含的数据量为Dv，计算出Dv的信息熵，给每个分支节点赋予Dv/D的权重，即样本分支节点的信息熵越大，它获得的信息增益越大。具体信息增益计算方法为  
![image](http://latex.codecogs.com/gif.latex?Gain(D,a)=Ent(D)-\sum_{v=1}^{V}\frac{|D_{v}|}{|D|}Ent(D^v))  

## ch4:朴素贝叶斯  
条件概率公式  
![image](http://latex.codecogs.com/gif.latex?p(c|x) = \frac{p(x|c)p(c)}{p(x)})  
如果p(c1|x,y)>p(c2|x,y) 那么分类就是c1，反之亦然




