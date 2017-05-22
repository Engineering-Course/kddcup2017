import numpy as np


def load_train_travel_data(batch_files):
    x_ = []
    y_ = []
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/training_travel_data/travel_{}.txt'.format(batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        item_y = [item_x[7]]
        del item_x[7]
        x_.append(item_x)
        y_.append(item_y)

    return np.array(x_), np.array(y_)


def load_test_travel_data(batch_files):
    x_ = []
    for id_ in xrange(len(batch_files)):
        filename = 'dataset/dataSets/testing_travel_data/travel_{}.txt'.format(batch_files[id_][:-1])
        with open(filename, 'r') as fr:
            line_ = fr.readline()
        data_ = line_.split(' ')
        item_x = [float(tt) for tt in data_]
        x_.append(item_x)

    return np.array(x_)



