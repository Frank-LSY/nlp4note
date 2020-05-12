import re
import time
import os
import pickle
import jsonlines
import numpy as np
# a = '[NOTES]20120112[DISCHARGE_SUMMARY].txt'
# b = re.sub(r'\[\w+\]','',a)
# a = a.strip('.txt').replace(']','').replace('[','')
# a= re.sub(r'\d{8}','-',a)
# print(a)
# print(b)

# with open('../stat/output2.jsonl','r') as f:
#     for item in jsonlines.Reader(f):
#         print('linex_index: ',item['linex_index'])
#         # print('token: ', item['features'][0]['token'])
#         print('layers: ', item['features'][0]['layers'][0]['index'])
#         print('len_val: ', len(item['features'][0]['layers'][0]['values']))
#         # print(item)
#         # time.sleep(0.5)

# def get_pt(file_in):
#     a = os.listdir(file_in)
#     with open('/data02/shuyu/stat/pt_list.pkl','wb') as f:
#         pickle.dump(a,f)
#     # print(a)

if __name__ == '__main__':
    with open('../bert_embedding/NOTES-DISCHARGE_SUMMARY/1220000159/20140926.jsonl', 'r') as f:
        for item in jsonlines.Reader(f):
            # a = np.array(item['features'][0]['layers'][-1]['values'])
            # print(item['features'][0]['layers'])
            print(item['features'][0]['layers'][-1]['values'])

            # print(len(item['features'][0]['layers'][-1]))
            print('linex_index (chunk#): ',item['linex_index'])
            print('First token for chunk#: ', item['features'][1]['token'])
            print('embedding_size: [{}×1]'.format(len(item['features'][0]['layers'][-1]['values'])))
            print('layers: ', item['features'][0]['layers'][0]['index'])
            print('[CLS] token: ', item['features'][0]['token'])
            print('embedding_size: [{}×1]'.format(len(item['features'][0]['layers'][-1]['values'])),'\n')
            # print('len_val: ', len(item['features'][0]['layers'][0]['values']))
            # print('val: ', item['features'][0]['layers'][0]['values'])
            # print(a)
            # print(len(a))