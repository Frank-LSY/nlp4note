import os
import pickle

if __name__ == '__main__':
    father_dir = '/home/shl183/nlp4note'
    types = os.listdir('{}/combine'.format(father_dir))
    pt_all = {}
    for type in types:
        print(type)
        with open('{}/combine/{}'.format(father_dir, type), 'rb') as f:
            sub_pt = pickle.load(f)
        for key, value in sub_pt.items():
            if key not in pt_all:
                pt_all[key] = [value]
            else:
                pt_all[key].append(value)
    with open('{}/stat/pt_all.pkl'.format(father_dir), 'wb') as f:
        pickle.dump(pt_all, f)
    # father_dir = '/home/shl183/nlp4note'
    # types = os.listdir('{}/pt_list'.format(father_dir))
    # for type in types:
    #     with open('{}/pt_list/{}'.format(father_dir,type),'rb') as f:
    #         ptlist = pickle.load(f)
    #     with open('{}/file/{}'.format(father_dir,type),'rb') as f:
    #         file_embedding = pickle.load(f)
    #
    #     pt_dict = {}
    #     for i in range(len(ptlist)):
    #         pt_dict[ptlist[i]] = file_embedding[i]
    #
    #     with open('{}/combine/{}'.format(father_dir,type),'wb') as f:
    #         pickle.dump(pt_dict,f)