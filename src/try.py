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
                    count += 1

    print(count)
