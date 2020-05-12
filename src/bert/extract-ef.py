import os
import re


if __name__ == '__main__':
    ef_dir = '/data02/shuyu/classified_txt/PROC_NOTES-MUGA_CARDIAC_EF'
    patients = os.listdir(ef_dir)
    ef_pt = {}
    count = 0
    for pat in patients:
        files = os.listdir('{}/{}'.format(ef_dir,pat))
        for file in files:
            with open ('{}/{}/{}'.format(ef_dir,pat,file),'r') as f:
                txt = f.read()
                # print(txt)
                ef = re.findall(r'\d+[.]\d+%|\d+%',txt)
                # print(ef)
                if len(ef)>0:
                    ef_pt['{}-{}'.format(pat,file[:-4])] = ef[-1]
                else:
                    count += 1
                    print(txt)

    print(count)
    with open('/data02/shuyu/stat/ef.csv','w') as f:
        print(ef_pt)
        for key,value in ef_pt.items():
            f.write('{},{}\n'.format(key,value))