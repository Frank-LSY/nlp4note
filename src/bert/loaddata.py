import numpy as np
import jsonlines
import os
import pickle

def get_plain_val(file_in):
    file_arr = []
    with open(file_in,'r') as f:
        for item in jsonlines.Reader(f):
            file_arr.append(np.array(item['features'][0]['layers'][-1]['values']))
    return np.array(file_arr)


if __name__ == '__main__':
    father_dir = '/data02/shuyu//bert_embedding/'
    # pt_list = '/data02/shuyu/stat/pt_list.pkl'
    outdir = '/data/shuyu//file_embedding'
    types = os.listdir(father_dir)
    for type in types:
        i = 0
        count = 0
        file_embed = []
        patients = os.listdir('{}/{}'.format(father_dir,type))
        for patient in patients:
            files = os.listdir('{}/{}/{}'.format(father_dir,type,patient))
            for file in files:
                try:
                    file_embed.append([patient,get_plain_val("{}/{}/{}/{}".format(father_dir,type,patient,file))])
                except:
                    continue
                i += 1
                print("{}\t{}/{}/{}".format(i,type,patient,file))
                if i==10000:
                    i = 0
                    count += 1
                    with open('{}/{}-{}.pkl'.format(outdir,type,count),'wb') as f:
                        pickle.dump(np.array(file_embed),f)
                        file_embed = []

        with open('{}/{}-{}.pkl'.format(outdir, type, count), 'wb') as f:
            pickle.dump(np.array(file_embed), f)
            file_embed = []