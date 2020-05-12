import os
import re
import shutil

if __name__ == '__main__':
    parent_archive = '../pt_df/'
    patients = os.listdir(parent_archive)
    count = 0
    for patient in patients:
        print(patient)
        if patient.isdigit() == False:
            continue
        else:
            files = os.listdir("{}/{}".format(parent_archive,patient))
            for file in files:
                if file[-3:] != 'txt':
                    continue
                else:
                    # print(file)
                    count += 1
                    # category = file.strip('.txt').replace(']','').replace('[','')
                    # category = re.sub(r'\d{8}','-',category)
                    # new_file = re.sub(r'\[\w+\]', '', file)
                    # dirs = "../classified_txt/{}/{}".format(patient,category)
                    dirs = "../classified_txts/{}".format(patient)
                    if not os.path.exists(dirs):
                        os.makedirs(dirs)
                    shutil.copy("{}/{}/{}".format(parent_archive,patient,file),"{}/{}".format(dirs,file))