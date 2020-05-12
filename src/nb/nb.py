import pickle
import csv
import numpy as np

def createVocabList(dataSet):  #创建词库 这里就是直接把所有词去重后，当作词库
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):  #文本词向量。词库中每个词当作一个特征，文本中就该词，该词特征就是1，没有就是0
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        # else:
        #     print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords) #防止某个类别计算出的概率为0，导致最后相乘都为0，所以初始词都赋值1，分母赋值为2.
    p1Num = np.ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  #这里使用了Log函数，方便计算，因为最后是比较大小，所有对结果没有影响。
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1): #比较概率大小进行判断，
    p1 = np.sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = np.sum(vec2Classify*p0Vec)+np.log(1-pClass1)
    if p1>p0:
        return True
    else:
        return False

def accuracy(tag,real):
    count = 0
    n = len(tag)
    for i in range(len(tag)):
        if tag[i]==real[i]:
            count += 1

    return count/n


if __name__ == '__main__':
    # with open('../../stat/corpus.txt', 'rb') as f:
    #     X = pickle.load(f)
    # with open ('../../stat/dis_pats_tag.csv','r') as f:
    #     reader = csv.reader(f)
    #     Y = [row[0] for row in reader]
    # for i in range(len(Y)):
    #     if Y[i]=='TRUE':
    #         Y[i]=1
    #     else:
    #         Y[i]=0
    # with open('../../stat/bert_nb.pkl','rb') as f:
    #     X,Y = pickle.load(f)

    # for i in range(len(Y)):
    #     Y[i] = int(Y[i])

    # for i in range(len(X)):
    #     X[i] = X[i].split()

    # x_train = X[:42000]
    # x_test = X[42000:]
    # y_train = Y[:42000]
    # y_test = Y[42000:]
    # # print(Y)

    # print('Construct Vocab')
    # vocablist = createVocabList(X)

    # print('Construct dataset')
    # trainMat=[]
    # i = 0
    # for postinDoc in x_train:
    #     i += 1
    #     if i%50==0:
    #         print(i)
    #     trainMat.append(setOfWords2Vec(vocablist, postinDoc))
    # testMat = []
    # i = 0
    # for postinDoc in x_test:
    #     i += 1
    #     if i%50==0:
    #         print(i)
    #     testMat.append(setOfWords2Vec(vocablist, postinDoc))
    # with open('./trainmat.pkl','wb') as f:
    #     pickle.dump(trainMat,f)
    # with open('./testmat.pkl','wb') as f:
    #     pickle.dump(testMat,f)
    with open('./trainmat.pkl','rb') as f:
        trainMat = pickle.load(f)
    with open('./testmat.pkl','rb') as f:
        testMat = pickle.load(f)
    print('Training')
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(y_train))
    print('Testing')
    train_tag = []
    for postinDoc in trainMat:
        res = classifyNB(np.array(postinDoc),p0V,p1V,pAb)
        train_tag.append(res)

    test_tag = []
    for postinDoc in testMat:
        res = classifyNB(np.array(postinDoc),p0V,p1V,pAb)
        test_tag.append(res)

    with open('./weight.wt','w') as f:
        f.write('{}\n'.format(p0V))
        f.write('{}\n'.format(p1V))
        f.write('{}\n'.format(pAb))

    with open('./train.tag','w') as f:
        for item in train_tag:
            f.write('{}\n'.format(item))

    with open('./test.tag','w') as f:
        for item in test_tag:
            f.write('{}\n'.format(item))

    with open('./metrics.txt','w') as f:
        f.write('train:{}\n'.format(accuracy(train_tag,y_train)))
        f.write('test:{}\n'.format(accuracy(test_tag, y_test)))