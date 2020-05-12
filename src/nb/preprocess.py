import csv
import numpy as np
import time
import pickle

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet,stopwords

def read_table(file_in):
    with open(file_in,'r') as f:
        reader = csv.DictReader(f)
        raw_X = []
        raw_Y = []
        id = []
        for row in reader:
            raw_X.append(row['PATH'])
            raw_Y.append(row['LEVEL'])
            id.append(row['GROUP_ID'])
    # print(raw_Y)
    X_path = []
    Y = []
    n = len(id)
    tmp_x = []
    for i in range(n-1):
        if raw_X[i][5] == '0':
            new_X= '/data02/shuyu/classified_txt/{}.txt'.format(raw_X[i][29:-6])
        else:
            new_X = '/data02/shuyu/classified_txt/{}.txt'.format(raw_X[i][27:-6])
        # print(new_X)
        # time.sleep(0.8)
        tmp_x.append(new_X)
        if raw_Y[i] != '':
            tmp_y = raw_Y[i]
        if id[i] == id[i+1]:
            continue
        else:
            X_path.append(tmp_x)
            Y.append(tmp_y)
            tmp_x = []

    # print(Y)
    print(len(Y))
    print(len(X_path))
    return X_path,Y

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def clean_txt(X_path):
    stopword = stopwords.words('english')
    corpus = []
    i = 0
    for case in X_path:
        sent = ''
        for sample in case:
            i += 1
            print(i, sample)
            with open('{}'.format(sample)) as f:
                content = f.readlines()
                sentence = ''

                for item in content:
                    sentence += item
                tokens = word_tokenize(sentence)  # 分词
                # print(tokens)
                new_tokens = list(set(tokens).difference(set(stopword)))
                # print(new_tokens)
                tagged_sent = pos_tag(new_tokens)  # 获取单词词性

                wnl = WordNetLemmatizer()
                lemmas_sent = []
                for tag in tagged_sent:
                    wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                    lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原

                for item in lemmas_sent:
                    sent += item + ' '
        corpus.append(sent)
    print(len(corpus))
    return corpus

if __name__ == '__main__':
    table_in = '../../stat/valid_data_labels_by_episode_30_days.csv'
    X_path,Y = read_table(table_in)
    X = clean_txt(X_path)
    with open('../../bert_nb.pkl','wb') as f:
        pickle.dump([X,Y],f)