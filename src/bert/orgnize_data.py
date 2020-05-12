import os
import pickle
if __name__ == '__main__':
    file_in = "/home/shl183/nlp4note/file_embedding/"
    patients = os.listdir(file_in)

    res_arr = []
    for patient in patients:
        print(patient)
        files = os.listdir("{}/{}".format(file_in,patient))
        pt_arr = [patient]
        for file in files:
            with open('{}/{}/{}'.format(file_in,patient,file),'rb') as f:
                a = pickle.load(f)
                pt_arr.append(a)
        res_arr.append(pt_arr)

    with open("/home/shl183/nlp4note/stat/pt_embedding.pkl",'wb') as f:
        pickle.dump(res_arr,f)