import numpy as np
import jsonlines
import os
import pickle
import sys

def get_plain_val(file_in):
    file_arr = []
    with open(file_in,'r') as f:
        for item in jsonlines.Reader(f):
            file_arr.append(np.array(item['features'][0]['layers'][-1]['values']))
    return np.array(file_arr)

def get_sum_value(file_in):
    sum_arr = np.zeros(768)
    count = 0
    with open(file_in,'r') as f:
        for item in jsonlines.Reader(f):
            count += 1
            sum_arr += np.array(item['features'][0]['layers'][-1]['values'])
    return sum_arr / count

if __name__ == '__main__':
    file_in = sys.argv[1]
    file_out = sys.argv[2]

    a = get_sum_value(file_in)
    # print(len(a))
    # print(a)

    with open(file_out, 'wb') as f:
        pickle.dump(np.array(a), f)
        file_embed = []