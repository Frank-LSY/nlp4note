import os
import sys
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


class Lstm():
    def __init__(self):
        self.model = Sequential()

    def parse_data(self,file_arr):
        tmp_X = []
        X = []
        Y = []
        max_len = 0
        pt_list = []
        for item in file_arr:
            if len(item[1])==0:
                continue
            else:
                pt_list.append(item[0])
                max_len = max(max_len,len(item[1])-1)
                Y.append(item[1][-1])
                tmp_X.append(item[1][:-1])

        for item in tmp_X:
            zero_arr = np.zeros([max_len-len(item),768])
            padded_item = np.vstack((item,zero_arr))
            # print(padded_item)
            X.append(padded_item)
        print(len(X))
        X = np.array(X)
        Y = np.array(Y)
        X = np.reshape(X, (len(X), max_len, 768))
        Y = np.reshape(Y, (len(X), 768))

        return X,Y,max_len,pt_list



    def padded(self,arr_in,max_len):
        zero_arr = np.zeros([max_len-len(arr_in),768])
        new_arr = np.vstack((arr_in,zero_arr))
        return new_arr

    def lstm(self,weight_dir,X,Y,max_len):

        self.model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, input_shape=(max_len, 768)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(768, activation='sigmoid'))
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(X, Y, epochs=100, batch_size=1024)
        self.model.save(weight_dir)


    def predict_next(self,arr_in):
        y = self.model.predict(arr_in)
        return y

if __name__ == '__main__':
    file_in = sys.argv[1]

    with open(file_in,'rb') as f:
        a= pickle.load(f)
        # print(a)
    print(len(a))
    lstm = Lstm()
    b_X, b_Y, max_len,pt_list= Lstm().parse_data(a)
    c = lstm.lstm('../weight/{}.h5'.format(file_in[18:4]), b_X, b_Y, max_len)
    d = lstm.parse_data(a)[0]
    e = lstm.predict_next(b_X)
    print(len(e[0]))
    with open('../pt_list/{}'.format(file_in[18:]),'wb') as f:
        pickle.dump(pt_list,f)

    with open('../file/{}'.format(file_in[18:]),'wb') as f:
        pickle.dump(e,f)


