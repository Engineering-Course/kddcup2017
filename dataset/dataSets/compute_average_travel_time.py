import os
import numpy as np

DATA_ROOT = 'testing_six_miss'
for root, dirs, files in os.walk(DATA_ROOT):
    for fname in files:
        print fname
        test_file = os.path.join(root, fname)
        with open(test_file, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        print data_
        x_ = [float(data_[tt]) for tt in range(1,7)]
        print x_
        non_zero = [dd for dd in x_ if dd != 0]
        print non_zero
        ave = np.mean(non_zero)
        print ave
        with open('testing_six_filled/all_ave/{}.txt'.format(fname[:-9]), 'w') as fw:
            fw.write(data_[0])
            for ii in range(1,len(data_)):
                if data_[ii] != str(0):
                    fw.write(' ' + data_[ii])
                else:
                    fw.write(' ' + str(ave))
        # wait = raw_input()

       

        
        
