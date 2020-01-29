import re
import jsonlines
import json
import time
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