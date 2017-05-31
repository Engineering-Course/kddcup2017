import numpy as np
import os


DATA_DIR = './result_combine/testing_six_stage5'

for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        test_file = os.path.join(root, fname)
        x_ = []
        with open(test_file, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        data_ = data_[:-1]
        item_x = [float(tt) for tt in data_]
        x_.append(item_x)

        with open('result_combine/stage_6/one/{}'.format(fname), 'r') as fi:
            l1 = fi.readline()
        r1 = float(l1)
        with open('result_combine/stage_6/two/{}'.format(fname), 'r') as fi:
            l2 = fi.readline()
        r2 = float(l2)
        with open('result_combine/stage_6/three/{}'.format(fname), 'r') as fi:
            l3 = fi.readline()
        r3 = float(l3)
        with open('result_combine/stage_6/four/{}'.format(fname), 'r') as fi:
            l4 = fi.readline()
        r4 = float(l4)
        with open('result_combine/stage_6/five/{}'.format(fname), 'r') as fi:
            l5 = fi.readline()
        r5 = float(l5)
        with open('result_combine/stage_6/six/{}'.format(fname), 'r') as fi:
            l6 = fi.readline()
        r6 = float(l6)

        res = (r1 * 1.0 + r2 * 2.0 + r3 * 3.0 + r4 * 4.0 + r5 * 5.0 + r6 * 6.0) / 21.0 
        # print res
        # print x_
        for jj in xrange(1, 6):
            x_[0][jj] = x_[0][jj+1]
        x_[0][6] = res
        print x_
        with open('result_combine/testing_six_final/{}'.format(fname), 'w') as fw:
            for ii in xrange(1, len(x_[0])-2):
                fw.write(str(x_[0][ii]) + ' ')

        # wait = raw_input()
