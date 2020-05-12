#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:00:47 2019

@author: frank-lsy
"""

import os
import csv
import shutil
import time
import pickle

if __name__ == '__main__':
    with open('../stat/[RE_ADMISSION].labels','r') as f:
        id = []
        date = []
        tag = []
        reader = csv.DictReader(f)
        for row in reader:
            id.append(row['STUDY_ID'])
            date.append(row['DATE'])
            tag.append(row['VALUE'])
    print(len(id))

    no_dis_pat = []
    dis_pat = []
    val_pat_dir = "../header/discharge/val-header-lists/"
    if not os.path.isdir(val_pat_dir):
        os.makedirs(val_pat_dir)

    for j,item in enumerate(id):
        print("[NOTES]{}[DISCHARGE_SUMMARY].txt".format(date[j]))
        for i in range(6):

                dis_pat.append(tag[j])
                break
        else:
            no_dis_pat.append(tag[j])


    with open ('../stat/dis_pats_tag.csv','w') as g:
        for item in dis_pat:
            g.write('{}\n'.format(item))


