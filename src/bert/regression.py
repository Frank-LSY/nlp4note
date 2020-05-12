from __future__ import print_function
# import tensorflow as tf

import csv
import numpy as np
rng = np.random

import os
import pickle

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split,cross_val_score

import matplotlib.pyplot as plt




def get_ef(pt_list):
    # father_dir = '../bert_embedding/'
    # types = os.listdir('../bert_embedding/')
    # patient_list = []
    # for type in types:
    #     patients = os.listdir('{}/{}'.format('../bert_embedding',type))
    #     for patient in patients:
    #         patient_list.append(patient)
    patient_list = pt_list
    patient_list.sort()
    # print(patient_list)

    # print(len(patient_list))
    ef_list = {}
    with open('../stat/ef.labels','r') as f:
        a = csv.reader(f)
        for row in a:
            if row[3] in patient_list:
                if row[3] not in ef_list:
                    ef_list[row[3]] = [row]
                else:
                    ef_list[row[3]].append(row)
    pt_ef = {}
    for key,value in ef_list.items():
        pt_ef[key] = value[-1][1]
    # print(pt_ef)
    return pt_ef

def get_patient_value(patient_dict):
    avg_pt_dict = {}
    for key,value in patient_dict.items():
        pt_arr = np.zeros(768)
        count = 0
        for item in value:
            count += 1
            pt_arr += item
        pt_arr = pt_arr/count
        avg_pt_dict[key] = pt_arr

    return avg_pt_dict

def regression(embedding,label,feature = 0,learning_rate = 0.01,training_epochs = 1000,display_step = 50):
    data_X = []
    data_Y = []
    x = []
    patients = []
    for key,value in label.items():
        patients.append(key)
        data_Y.append(value)
    for patient in patients:
        data_X.append(embedding[patient])
        x.append(embedding[patient][feature])
    train_X = np.array(data_X)
    train_Y = np.array(data_Y)

    lin_regress = linear_model.LinearRegression()
    lin_regress.fit(train_X,train_Y)
    print('Coefficients:',lin_regress.coef_)
    print('Intercept: ',lin_regress.intercept_)

    return x,train_Y

if __name__ == '__main__':
    father_dir = '/home/shl183/nlp4note'
    with open('{}/stat/pt_all.pkl'.format(father_dir),'rb') as f:
        pt_dict = pickle.load(f)

    # print(pt_dict)

    embedding = get_patient_value(pt_dict)
    pt_list = []
    for key in embedding:
        pt_list.append(key)
    pt_ef = get_ef(pt_list)
    # with open('./patient_embedding.pkl','rb') as f:
    #     embedding = pickle.load(f)
    #
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 50

    dataset_x, dataset_y = regression(embedding,pt_ef,1,learning_rate,training_epochs,display_step)

    plt.scatter(dataset_x, dataset_y, color='orange')
    plt.show()