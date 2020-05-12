import pickle

with open('../../stat/corpus.txt','rb') as f:
    info = pickle.load(f)

with open('../../stat/ori_corpus.txt','w') as f:
    for item in info:
        f.write('{}\n'.format(item))
# print(info)