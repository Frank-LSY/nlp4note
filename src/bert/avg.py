import jsonlines
import time
import os
import numpy as np
import pickle


def get_sum_value(file_in):
    sum_arr = np.zeros(768)
    count = 0
    with open(file_in,'r') as f:
        for item in jsonlines.Reader(f):
            count += 1
            sum_arr += np.array(item['features'][0]['layers'][-1]['values'])
    return sum_arr / count

def get_patient_value(patient_id,value,file_in):
    patient_arr = []
    # print(patient_id)
    # print(value)
    for item in value:
        for it in item[1]:
            try:
                print(file_in,item[0],patient_id,it)
                file_embed = get_sum_value('{}/{}/{}/{}'.format(file_in,item[0],patient_id,it))
                patient_arr.append([it,file_embed])
            except jsonlines.jsonlines.InvalidLineError:
                continue
    return patient_arr


if __name__ == '__main__':
    father_dir = '/data02/shuyu/bert_embedding/'
    pt_list = '/data02/shuyu/stat/pt_list.pkl'
    outdir = '/data/shuyu/patient_embedding'
    types = os.listdir(father_dir)
    with open(pt_list,'rb') as f:
        patients = pickle.load(f)
    dummy_dict = {}
    for type in types:
        for patient in patients:
            try:
                file = os.listdir('{}/{}/{}'.format(father_dir, type, patient))
            except FileNotFoundError:
                continue
            if patient not in dummy_dict:
                dummy_dict[patient] = [[type,file]]
            else:
                dummy_dict[patient].append([type, file])
    # print(dummy_dict)
    pt_embedding ={}
    for k,v in dummy_dict.items():
        print(k)
        print(v)
        pt_embedding[k]=get_patient_value(k,v,father_dir)

    with open('{}/pt_all.pkl'.format(outdir), 'wb') as f:
        pickle.dump(pt_embedding, f)


