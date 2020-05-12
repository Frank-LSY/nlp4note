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

    p1Num = np.ones(numWords) #防止某个类别计算出的概率为0，导致最后相乘都为0，所以初始词都赋值1，分母赋值为2.
    p2Num = np.ones(numWords)
    p3Num = np.ones(numWords)
    p4Num = np.ones(numWords)
    p5Num = np.ones(numWords)
    p1Denom = 2
    p2Denom = 2
    p3Denom = 2
    p4Denom = 2
    p5Denom = 2
    pAb1 = 0
    pAb2 = 0
    pAb3 = 0
    pAb4 = 0
    pAb5 = 0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            pAb1 += 1
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        if trainCategory[i] == 2:
            pAb2 += 1
            p2Num += trainMatrix[i]
            p2Denom += np.sum(trainMatrix[i])
        if trainCategory[i] == 3:
            pAb3 += 1
            p3Num += trainMatrix[i]
            p3Denom += np.sum(trainMatrix[i])
        if trainCategory[i] == 4:
            pAb4 += 1
            p4Num += trainMatrix[i]
            p4Denom += np.sum(trainMatrix[i])
        if trainCategory[i] == 5:
            pAb5 += 1
            p5Num += trainMatrix[i]
            p5Denom += np.sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  #这里使用了Log函数，方便计算，因为最后是比较大小，所有对结果没有影响。
    p2Vect = np.log(p2Num / p2Denom)
    p3Vect = np.log(p3Num / p3Denom)
    p4Vect = np.log(p4Num / p4Denom)
    p5Vect = np.log(p5Num / p5Denom)
    pAbusive = np.array([pAb1,pAb2,pAb3,pAb4,pAb5]) / float(numTrainDocs)
    return p1Vect, p2Vect, p3Vect, p4Vect, p5Vect, pAbusive

def classifyNB(vec2Classify,p1Vec, p2Vec, p3Vec, p4Vec, p5Vec,pAbusive): #比较概率大小进行判断，
    p1 = np.sum(vec2Classify*p1Vec)+np.log(pAbusive[0])
    p2 = np.sum(vec2Classify*p2Vec)+np.log(pAbusive[1])
    p3 = np.sum(vec2Classify * p3Vec) + np.log(pAbusive[2])
    p4 = np.sum(vec2Classify * p4Vec) + np.log(pAbusive[3])
    p5 = np.sum(vec2Classify * p5Vec) + np.log(pAbusive[4])

    tmp = [p1,p2,p3,p4,p5]
    return tmp.index(max(tmp))

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
    with open('../../stat/bert_nb.pkl','rb') as f:
        X,Y = pickle.load(f)

    for i in range(len(Y)):
        Y[i] = int(Y[i])

    for i in range(len(X)):
        X[i] = X[i].split()

    x_train = X[:3300]
    x_test = X[3300:]
    y_train = Y[:3300]
    y_test = Y[3300:]
    # print(Y)

    # print('Construct Vocab')
    # vocablist = createVocabList(X)
    #
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
    p1V,p2V,p3V,p4V,p5V,pAb = trainNB0(np.array(trainMat),np.array(y_train))

    with open('./weight.wt','w') as f:
        f.write('{}\n'.format(p1V))
        f.write('{}\n'.format(p2V))
        f.write('{}\n'.format(p3V))
        f.write('{}\n'.format(p4V))
        f.write('{}\n'.format(p5V))
        f.write('{}\n'.format(pAb))

    print('Testing')
    train_tag = []
    for postinDoc in trainMat:
        res = classifyNB(np.array(postinDoc),p1V,p2V,p3V,p4V,p5V,pAb)
        train_tag.append(res)
    # print(train_tag)
    test_tag = []
    for postinDoc in testMat:
        res = classifyNB(np.array(postinDoc),p1V,p2V,p3V,p4V,p5V,pAb)
        test_tag.append(res)
    # print(test_tag)



    with open('./train.tag','w') as f:
        for item in train_tag:
            f.write('{}\n'.format(item))

    with open('./test.tag','w') as f:
        for item in test_tag:
            f.write('{}\n'.format(item))

    with open('./metrics.txt','w') as f:
        f.write('train:{}\n'.format(accuracy(train_tag,y_train)))
        f.write('test:{}\n'.format(accuracy(test_tag, y_test)))