# import numpy as np
import torch
import nltk
import os
# import time
import shutil
import pickle
from nltk.tokenize import sent_tokenize
# Load Model
from InferSent.models import InferSent

MODEL_PATH = "encoder/infersent1.pkl"
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# CUDA
use_cuda = True
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
print('Load glove')
model.build_vocab_k_words(K=2000000)

# load sentence
dis = '/home/shl183/nlp4note/classified_txt/discharge-sep'
res = '/home/shl183/nlp4note/infersent'
patients = os.listdir(dis)
# exist = os.listdir(res)
# with open('./tmp.pkl','wb') as f:
#     pickle.dump(exist,f)
for patient in patients:
    # if patient in exist:
    #     continue
    notes = os.listdir('{}/{}'.format(dis,patient))
    for note in notes:
        tps = os.listdir('{}/{}/{}'.format(dis,patient,note))
        if os.path.exists('{}/{}/{}'.format(res,patient,note)):
            shutil.rmtree('{}/{}/{}'.format(res,patient,note))
        os.makedirs('{}/{}/{}'.format(res,patient,note))
        for tp in tps:
            print('{}/{}/{}/{}'.format(dis,patient,note,tp))
            with open ('{}/{}/{}/{}'.format(dis,patient,note,tp),'r') as f:
                sents = f.read()
            t_sents = sent_tokenize(sents)
#                 print(len(t_sent))
#             time.sleep(0.4)
            val_sent = []
            with open ('{}/{}/{}/{}pkl'.format(res,patient,note,tp[:-3]),'wb') as f:
                for sent in t_sents:
                    # print(sent)
                    length = len(sent.split())
                    if length<10:
                        continue
                    val_sent.append(sent)
                #print(val_sent)
                if val_sent == []:
                    continue
                embedding = model.encode(val_sent, bsize=128, tokenize=False, verbose=True)
                pickle.dump(embedding,f)
