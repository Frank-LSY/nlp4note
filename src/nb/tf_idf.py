#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:00:47 2019

@author: frank-lsy
"""

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet,stopwords

from sklearn import feature_extraction

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import preprocessing

import os
import gc
import pickle

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



if __name__ == '__main__':

    local = '../../topic'
    linux = '/data02/shuyu/header/discharge/val-header-list/'
    stopword = stopwords.words('english')
    notes = os.listdir(linux)
    corpus = []
    i=0
    for note in notes:
        i += 1
        if i%1000==0:
            print('{} file.'.format(i))
        with open('{}/{}'.format(linux,note)) as f:
            content = f.readlines()
            sentence = ''
            sent = ''
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
                sent += item+' '
            corpus.append(sent)

    c_file = open('../stat/new_corpus.txt','wb')
    pickle.dump(corpus,c_file)


    print('TFIDF')
    tfidf_vec = TfidfVectorizer(ngram_range=(1,3))
    tfidf = tfidf_vec.fit_transform(corpus)
    t_file = open('../stat/new_tfidf.pkl','wb')
    pickle.dump(tfidf,t_file)

    del corpus
    gc.collect()

    # print(len(tfidf.toarray()[0]))
    print('NORMALIZATION')
    normed_tfidf = preprocessing.normalize(tfidf.toarray(),norm='l2')
    n_file = open('../stat/norm-tfidf.pkl','wb')
    pickle.dump(normed_tfidf,n_file)

    del tfidf
    gc.collect()

    pca = PCA(n_components=1000)
    print('PCA')
    pca_data = pca.fit_transform(normed_tfidf)
    p_file = open('../stat/pca.pkl','wb')
    pickle.dump(pca_data,p_file)

    del normed_tfidf
    gc.collect()

    min_max_scaler = preprocessing.MinMaxScaler()
    print('MINMAX')
    minMax_pca = min_max_scaler.fit_transform(pca_data)

    del pca_data
    gc.collect()

    print('PICKLE')
    data = {}
    for i,note in enumerate(notes):
        data['{}/{}'.format(linux,note)]= minMax_pca[i]

    g = open('../stat/tfidf-pca-topics.pkl', 'wb')
    pickle.dump(data, g)